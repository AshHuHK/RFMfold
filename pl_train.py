# train_lightning.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from polars.interchange.utils import polars_dtype_to_dtype_map
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
import pickle
from config import _C as cfg
import argparse
from yacs.config import CfgNode
import numpy as np

# --- Import user-defined modules ---
from rfmfold import RFMfold
from data import RNADataset, pad_collate

import logging
logger = logging.getLogger(__name__)

# --- Utility Functions ---
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
def convert_to_dict(cfg_node, key_list):
    def _assert_with_logging(cond, msg):
        if not cond:
            logger.debug(msg)
        assert cond, msg

    def _valid_type(value, allow_cfg_node=False):
        return (type(value) in _VALID_TYPES) or (
                allow_cfg_node and isinstance(value, CfgNode)
        )

    if not isinstance(cfg_node, CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict

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
        print(self.model)
        self.current_sample_ratio = sampler_config['start_ratio']
        self.validation_step_outputs = []
        self.save_ss_dir = None

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

        if adj is not None:
            tp, fp, fn, f1_list = calculate_f1_metrics(preds, adj.squeeze(1), mask.squeeze(1))
            self.validation_step_outputs.append({'tp': tp, 'fp': fp, 'fn': fn, 'f1_list': f1_list})
        else:
            raise Exception("No Available Labels for Validation Step!")

    def predict_step(self, batch, batch_idx):
        outer, energy, ss_fea, adj, mask = self._prepare_batch(batch)
        logits = self(outer, energy, ss_fea, mask).squeeze(1)

        logits = 0.5 * (logits + logits.transpose(1, 2))
        logits = logits - torch.diag_embed(torch.diagonal(logits, dim1=-2, dim2=-1))

        preds = (torch.sigmoid(logits) > 0.5).float()

        if self.save_ss_dir is not None:
            self.save_ss(batch, preds)

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
        #energy = batch["energy"]
        if batch["adj"] is not None:
            adj = batch["adj"].unsqueeze(1)
        else:
            adj = None
        
        if "ss_prob" in batch and batch["ss_prob"] is not None:
            ss_prob = batch["ss_prob"]
            ss_pred = (ss_prob > 0.5).float()
            ss_fea = torch.cat([ss_pred, ss_prob], dim=1)
        else:
            ss_fea = torch.empty(outer.shape[0], 0, *outer.shape[2:], device=self.device)
        return outer, energy, ss_fea, adj, mask

    def postprocess(self, prob_map, seq, threshold=0.5, allow_nc=True):
        # we suppose that probmay cantains values range from [0,1], so is the threshold
        canonical_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']

        prob_map = prob_map * (1 - np.eye(prob_map.shape[0]))  # no  care about the diagonal
        pred_map = (prob_map > threshold)

        seq_len = len(seq)
        x_array, y_array = np.nonzero(pred_map)
        prob_array = []
        for i in range(x_array.shape[0]):
            prob_array.append(prob_map[x_array[i], y_array[i]])
        prob_array = np.array(prob_array)

        sort_index = np.argsort(-prob_array)

        mask_map = np.zeros_like(pred_map)
        already_x = set()
        already_y = set()
        multiplet_list = []
        for index in sort_index:
            x = x_array[index]
            y = y_array[index]

            # # no sharp stem-loop
            if abs(x - y) <= 1:  # when <=1, allow 1 element loop
                continue

            seq_pair = seq[x] + seq[y]
            if seq_pair not in canonical_pairs and allow_nc == False:
                # print(seq_pair)
                continue
                pass

            if x in already_x or y in already_y:  # this is conflict
                multiplet_list.append([x + 1, y + 1])
                continue
            else:
                mask_map[x, y] = 1
                already_x.add(x)
                already_y.add(y)

        pred_map_without_multiplets = pred_map * mask_map

        return pred_map, pred_map_without_multiplets, multiplet_list

    def matrix2ct(self, prob_map, seq, seq_id, ct_dir, threshold=0.5, with_post=False, nc=False):
        """
        :param contact: binary matrix numpy
        :param seq: string
        :return:
        """
        # 1.process matrix to make it obey the required constraints (maybe need sequence string)
        if with_post == True:
            pred_map, pred_map_without_multiplets, multiplet_list = self.postprocess(
                prob_map, threshold=threshold, seq=seq, allow_nc=True
            )
            contact = pred_map_without_multiplets
        else:
            if threshold > 0:
                contact = (prob_map > threshold)
            else:
                contact = prob_map

        # 2.write ct file
        seq_len = len(seq)
        structure = np.where(contact)
        pair_dict = dict()
        for i in range(seq_len):
            pair_dict[i] = -1
        for i in range(len(structure[0])):
            pair_dict[structure[0][i]] = structure[1][i]
        first_col = list(range(1, seq_len + 1))
        second_col = list(seq)
        third_col = list(range(seq_len))
        fourth_col = list(range(2, seq_len + 2))
        fifth_col = [pair_dict[i] + 1 for i in range(seq_len)]
        last_col = list(range(1, seq_len + 1))

        if os.path.exists(ct_dir) != True:
            os.makedirs(ct_dir)
        ct_file = os.path.join(ct_dir, seq_id + ".ct")

        with open(ct_file, "w") as f:
            f.write("{}\t{}\n".format(seq_len, seq_id))  # header
            for i in range(seq_len):
                f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(first_col[i], second_col[i], third_col[i], fourth_col[i],
                                                          fifth_col[i], last_col[i]))


    def set_save_ss_dir(self, save_ss_dir: str, threshold: float=0.5, allow_nc: bool=True):
        self.save_ss_dir = save_ss_dir
        self.save_prob_dir = os.path.join(self.save_ss_dir, 'prob')
        os.makedirs(self.save_prob_dir, exist_ok=True)
        self.save_cm_fb_dir = os.path.join(self.save_ss_dir, 'cm_fb')
        os.makedirs(self.save_cm_fb_dir, exist_ok=True)
        self.save_cm_nm_dir = os.path.join(self.save_ss_dir, 'cm_nm')
        os.makedirs(self.save_cm_nm_dir, exist_ok=True)
        self.save_ct_dir = os.path.join(self.save_ss_dir, 'ct')
        os.makedirs(self.save_ct_dir, exist_ok=True)
        self.threshold = threshold
        self.allow_nc = allow_nc

    def save_ss(self, batch, preds):
        names = batch['names']
        seqs = batch['seq_str']
        lengths = batch['lengths']
        for i in range(len(names)):
            name = names[i]
            length = lengths[i]
            seq = seqs[i]
            prob_map = preds[i][:length, :length].cpu().numpy()

            np.save(os.path.join(self.save_prob_dir, "{}".format(name)), prob_map)

            # 1.with multiplets 2.without multiplets (can create ct, dot);
            post_map, post_map_nomlets, multiplet_list = self.postprocess(
                prob_map, seq, threshold=self.threshold, allow_nc=self.allow_nc
            )
            np.save(os.path.join(self.save_cm_fb_dir, "{}".format(name)), post_map)
            np.save(os.path.join(self.save_cm_nm_dir, "{}".format(name)), post_map_nomlets)

            # save ct file, with post_without_mbp_numpy
            self.matrix2ct(post_map_nomlets, seq, name, self.save_ct_dir, threshold=self.threshold, with_post=False, nc=self.allow_nc)

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
                energy_dict=self.energy_dict, energy_dist_dict=self.energy_dist_dict,
                active_methods=self.data_config['active_methods'],
            )
            self.val_dataset = RNADataset(
                root=self.data_config['val_root'],
                feature_parent_dir=self.data_config['feature_parent_dir']['val'],
                energy_dict=self.energy_dict, energy_dist_dict=self.energy_dist_dict,
                active_methods = self.data_config['active_methods'],
            )
        # Set up validation dataset when fitting OR validating
        if stage in ('fit', 'validate', 'predict') or stage is None:
            # We check for val_root to avoid errors
            if self.val_dataset is None and self.data_config.get('val_root'):
                print("--- Setting up Validation Dataset ---")
                self.val_dataset = RNADataset(
                    root=self.data_config['val_root'],
                    feature_parent_dir=self.data_config['feature_parent_dir'].get('val'),
                    energy_dict=self.energy_dict, energy_dist_dict=self.energy_dist_dict,
                    active_methods=self.data_config['active_methods'],
                )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=pad_collate, **self.loader_config['train'])
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=pad_collate, **self.loader_config['val'])

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, collate_fn=pad_collate, **self.loader_config['val'])

# ============================================================================
# 4. Main Execution
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Classification Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)  # nargs=argparse.REMAINDER是指所有剩余的参数均转化为一个列表赋值给此项
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    MODEL_CONFIG = convert_to_dict(cfg.MODEL, [])
    OPTIMIZER_CONFIG = convert_to_dict(cfg.OPTIMIZER, [])
    SCHEDULER_CONFIG = convert_to_dict(cfg.SCHEDULER, [])
    SAMPLER_CONFIG = convert_to_dict(cfg.SAMPLER, [])
    DATA_CONFIG = convert_to_dict(cfg.DATA, [])
    LOADER_CONFIG = convert_to_dict(cfg.LOADER, [])
    TRAINER_CONFIG = convert_to_dict(cfg.TRAINER, [])

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