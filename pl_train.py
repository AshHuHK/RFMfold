# train_lightning.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import pickle

# --- Import user-defined modules ---
from rfmfold_old import RFMfold
from data import RNADataset, pad_collate

# --- Utility Functions ---
def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def symmetric_masked_bce_loss_vectorized(logits, target, mask, sample_ratio=0.5):
    """ Fully vectorized symmetric BCE loss with negative sampling. """
    if logits.dim() == 4:
        logits, target, mask = logits.squeeze(1), target.squeeze(1), mask.squeeze(1)
    
    B, N, _ = logits.shape
    logits = 0.5 * (logits + logits.transpose(-2, -1))

    idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=logits.device)
    
    logits_flat = logits[:, idx_i, idx_j]
    target_flat = target[:, idx_i, idx_j]
    mask_flat   = mask[:, idx_i, idx_j]

    valid = mask_flat > 0
    pos_mask = (target_flat == 1) & valid
    neg_mask = (target_flat == 0) & valid

    num_neg_per_sample = neg_mask.sum(dim=1)
    num_samp_per_sample = (sample_ratio * num_neg_per_sample).to(torch.long)

    rand_for_sampling = torch.rand_like(neg_mask, dtype=torch.float32)
    rand_for_sampling[~neg_mask] = -1.0
    
    _, sorted_indices = torch.sort(rand_for_sampling, dim=1, descending=True)

    topk_mask = torch.arange(sorted_indices.shape[1], device=logits.device) < num_samp_per_sample.unsqueeze(1)
    
    sampled_neg_mask = torch.zeros_like(neg_mask)
    sampled_neg_mask.scatter_(dim=1, index=sorted_indices, src=topk_mask)

    keep_mask = pos_mask | sampled_neg_mask
    
    if not keep_mask.any():
        return torch.tensor(0.0, device=logits.device)
        
    final_logits = logits_flat[keep_mask]
    final_target = target_flat[keep_mask]
    
    loss = F.binary_cross_entropy_with_logits(final_logits, final_target, reduction='mean')
    return loss

def calculate_f1_metrics(preds, labels, masks):
    """ Helper to compute TP, FP, FN, and F1 scores per sample. """
    TP, FP, FN = 0, 0, 0
    f1_list = []
    for b in range(preds.size(0)):
        p, l, m = preds[b], labels[b], masks[b]
        valid = m > 0
        tp = ((p == 1) & (l == 1) & valid).sum().item()
        fp = ((p == 1) & (l == 0) & valid).sum().item()
        fn = ((p == 0) & (l == 1) & valid).sum().item()
        TP += tp; FP += fp; FN += fn
        prec_b = tp / (tp + fp + 1e-8); rec_b = tp / (tp + fn + 1e-8)
        f1_b = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-8)
        f1_list.append(f1_b)
    return TP, FP, FN, f1_list

# ============================================================================
# 2. LightningModule
# ============================================================================
class RNASegmenter(pl.LightningModule):
    def __init__(self, model_config: dict, optimizer_config: dict, scheduler_config: dict, sampler_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = RFMfold(**model_config)
        self.current_sample_ratio = sampler_config['start_ratio']
        self.validation_step_outputs = []
        
    def forward(self, x_main, energy, ss_fea, mask):
        return self.model(x_main, energy, ss_fea, mask)

    def on_train_epoch_start(self):
        sampler_cfg = self.hparams.sampler_config
        epoch = self.current_epoch
        calculated_ratio = sampler_cfg['start_ratio'] + sampler_cfg['step_factor'] * (epoch // sampler_cfg['step_every_n_epochs'])
        self.current_sample_ratio = min(sampler_cfg['end_ratio'], calculated_ratio)
        self.log('sample_ratio', self.current_sample_ratio, prog_bar=False)

    def training_step(self, batch, batch_idx):
        outer, energy, ss_fea, adj, mask = self._prepare_batch(batch)
        logits = self(outer, energy, ss_fea, mask)
        loss = symmetric_masked_bce_loss_vectorized(logits, adj, mask, sample_ratio=self.current_sample_ratio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outer, energy, ss_fea, adj, mask = self._prepare_batch(batch)
        logits = self(outer, energy, ss_fea, mask).squeeze(1)
        
        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits - torch.diag_embed(torch.diagonal(logits, dim1=-2, dim2=-1))
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        
        tp, fp, fn, f1_list = calculate_f1_metrics(preds, adj.squeeze(1), mask.squeeze(1))
        self.validation_step_outputs.append({'tp': tp, 'fp': fp, 'fn': fn, 'f1_list': f1_list})

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if not outputs: return
            
        total_tp = sum(x['tp'] for x in outputs)
        total_fp = sum(x['fp'] for x in outputs)
        total_fn = sum(x['fn'] for x in outputs)
        all_f1s = [item for x in outputs for item in x['f1_list']]

        micro_prec = total_tp / (total_tp + total_fp + 1e-8)
        micro_rec = total_tp / (total_tp + total_fn + 1e-8)
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec + 1e-8)
        macro_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

        self.log('val_micro_f1', micro_f1, prog_bar=True, sync_dist=True)
        self.log('val_macro_f1', macro_f1, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_cfg = self.hparams.optimizer_config
        sch_cfg = self.hparams.scheduler_config
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_cfg['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_cfg['step_size'], gamma=sch_cfg['gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _prepare_batch(self, batch):
        outer = batch["seq_outer"].permute(0, 3, 1, 2)
        mask = batch["mask"].unsqueeze(1)
        energy = torch.tanh(batch["energy"])
        adj = batch["adj"].unsqueeze(1)
        
        if "ss_prob" in batch and batch["ss_prob"] is not None:
            ss_prob = batch["ss_prob"]
            ss_pred = (ss_prob > 0.5).float()
            ss_fea = torch.cat([ss_pred, ss_prob], dim=1)
        else:
            ss_fea = torch.empty(outer.shape[0], 0, *outer.shape[2:], device=self.device)
        return outer, energy, ss_fea, adj, mask

# ============================================================================
# 3. LightningDataModule
# ============================================================================
class RNADataModule(pl.LightningDataModule):
    def __init__(self, data_config: dict, loader_config: dict):
        super().__init__()
        self.data_config = data_config
        self.loader_config = loader_config
        self.energy_dict = load_from_pickle(data_config['energy_dict_path'])
        self.energy_dist_dict = load_from_pickle(data_config['energy_dist_dict_path'])
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str):
        if stage == "fit" and self.train_dataset is None:
            print("--- Setting up Train/Val Datasets ---")
            self.train_dataset = RNADataset(
                root=self.data_config['train_root'],
                feature_parent_dir=self.data_config['feature_parent_dir']['train'],
                energy_dict=self.energy_dict, energy_dist_dict=self.energy_dist_dict
            )
            self.val_dataset = RNADataset(
                root=self.data_config['val_root'],
                feature_parent_dir=self.data_config['feature_parent_dir']['val'],
                energy_dict=self.energy_dict, energy_dist_dict=self.energy_dist_dict
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=pad_collate, **self.loader_config['train'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=pad_collate, **self.loader_config['val'])

# ============================================================================
# 4. Main Execution
# ============================================================================
def main():
    # --- Centralized Configurations ---
    MODEL_CONFIG = {
        "main_ch": 16, "energy_ch": 2, "ss_fea_ch": -1, # ss_fea_ch is set dynamically
        "base_ch": 128, "depth": 6, "drop_p": 0.15,
        "dilations": (1, 2, 4, 8, 16)
    }
    OPTIMIZER_CONFIG = {"lr": 1e-4}
    SCHEDULER_CONFIG = {"step_size": 10, "gamma": 0.5}
    SAMPLER_CONFIG = {
        "start_ratio": 0.5, "end_ratio": 0.6, 
        "step_factor": 0.01, "step_every_n_epochs": 2
    }
    DATA_CONFIG = {
        "train_root": "/workspace/ash/DAT/bprna/TR0",
        "val_root": "/workspace/ash/DAT/bprna/TS0",
        "energy_dict_path": "/workspace/ash/code/RNA-FM/ash/avg_energy_stacking_k2.pkl",
        "energy_dist_dict_path": "/workspace/ash/code/RNA-FM/ash/avg_energy_dist_k2.pkl",
        "feature_parent_dir": {
            "train": "/workspace/ash/code/RNA0730/features/TR0", 
            "val": "/workspace/ash/code/RNA0730/features/TS0"
        }
    }
    LOADER_CONFIG = {
        "train": {"batch_size": 8, "shuffle": True, "num_workers": 4},
        "val": {"batch_size": 8, "shuffle": False, "num_workers": 4}
    }
    TRAINER_CONFIG = {
        "max_epochs": 100,
        "accelerator": "gpu",
        "devices": "auto",
        "strategy": "ddp",
        "precision": "16-mixed",
        "log_every_n_steps": 10
    }

    # --- Initialization and Dynamic Configuration ---
    pl.seed_everything(3407, workers=True)
    torch.set_float32_matmul_precision('high')

    data_module = RNADataModule(DATA_CONFIG, LOADER_CONFIG)
    data_module.setup('fit')

    num_feature_methods = len(data_module.train_dataset.feature_methods)
    print(f"Dynamically detected {num_feature_methods} feature methods: {data_module.train_dataset.feature_methods}")
    
    # Dynamically set channel numbers
    MODEL_CONFIG['ss_fea_ch'] = num_feature_methods * 2
    
    print(f"Model config updated with ss_fea_ch = {MODEL_CONFIG['ss_fea_ch']}")
    
    model_module = RNASegmenter(MODEL_CONFIG, OPTIMIZER_CONFIG, SCHEDULER_CONFIG, SAMPLER_CONFIG)
    
    # --- Callbacks and Trainer Setup ---
    checkpoint_callback = ModelCheckpoint(
        monitor='val_macro_f1',
        dirpath='checkpoints_lightning/',
        filename='rna-segmenter-{epoch:02d}-{val_macro_f1:.4f}',
        save_top_k=3,
        mode='max',
    )
    early_stop_callback = EarlyStopping(
        monitor='val_macro_f1', 
        patience=15,
        mode='max'
    )
    progress_bar = TQDMProgressBar(refresh_rate=5)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        **TRAINER_CONFIG
    )
    
    print("🚀 Starting training with dynamically configured model... 🚀")
    trainer.fit(model_module, datamodule=data_module)

if __name__ == "__main__":
    main()