# pipeline_inference.py

import os
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, os.path.abspath('.')) # 

from rfmfold import RFMfold
from data import read_fasta, seq_to_onehot, build_symmetric_energy_matrix
from ss_generators import generate_all_ss_features # <

# ==============================================================================
# STAGE 2: RFMfold 推理 (这部分逻辑保留在主文件中)
# ==============================================================================
def load_from_pickle(path):
    with open(path, "rb") as f: return pickle.load(f)

def load_model_from_checkpoint(config: dict, ckpt_path: str, device: torch.device) -> RFMfold:
    print(f"--- Loading RFMfold from checkpoint: {ckpt_path} ---")
    model = RFMfold(**config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get('state_dict')
    if state_dict is None: raise KeyError("Checkpoint does not contain 'state_dict'.")
    if next(iter(state_dict)).startswith("model."):
        new_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("RFMfold model loaded successfully.")
    return model

def load_ss_features(feature_parent_dir: str, sample_name: str, seq_len: int) -> torch.Tensor:
    parent_path = Path(feature_parent_dir)
    if not parent_path.exists(): return None
    all_ss_feats = []
    feature_methods = sorted([d.name for d in parent_path.iterdir() if d.is_dir()])
    for method in feature_methods:
        npy_path = parent_path / method / f"{sample_name}.npy"
        if npy_path.exists():
            arr = np.load(npy_path)
            if arr.ndim != 2 or arr.shape[0] != seq_len or arr.shape[1] != seq_len: continue
            feat_tensor = torch.from_numpy(arr).float()
            if torch.any(feat_tensor > 1.0) or torch.any(feat_tensor < 0.0): feat_tensor = torch.sigmoid(feat_tensor)
            all_ss_feats.append((feat_tensor > 0.5).float().unsqueeze(0))
            all_ss_feats.append(feat_tensor.unsqueeze(0))
    return torch.cat(all_ss_feats, dim=0) if all_ss_feats else None

def run_stage2_rfmfold_inference(args):
    """
    (Stage 2) 使用第一阶段生成的特征，运行主RFMfold模型。
    """
    print("\n==============================================")
    print("= STAGE 2: Running Main RFMfold Inference    =")
    print("==============================================")
    device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    seq = read_fasta(args.fasta_file)
    seq_len = len(seq)
    sample_name = Path(args.fasta_file).stem
    
    onehot = seq_to_onehot(seq)
    energy_dict = load_from_pickle(args.energy_dict_path)
    energy_dist_dict = load_from_pickle(args.energy_dist_dict_path)
    energy = build_symmetric_energy_matrix(seq, energy_dict, energy_dist_dict)
    ss_fea = load_ss_features(args.ss_feature_dir, sample_name, seq_len)
    num_ss_fea_channels = ss_fea.shape[0] if ss_fea is not None else 0
    
    MODEL_CONFIG = {
        "main_ch": 16, "energy_ch": 2, "ss_fea_ch": num_ss_fea_channels,
        "base_ch": 128, "depth": 6, "drop_p": 0.0,
        "dilations": (1, 2, 4, 8, 16)
    }
    print(f"Initializing RFMfold with config: {MODEL_CONFIG}")
    model = load_model_from_checkpoint(MODEL_CONFIG, args.rfmfold_checkpoint_path, device)
    
    seq_outer = torch.einsum('if,jg->ijfg', onehot, onehot).reshape(1, seq_len, seq_len, 16).permute(0, 3, 1, 2).to(device)
    energy_tensor = torch.tanh(energy.unsqueeze(0)).permute(0, 3, 1, 2).to(device)
    ss_fea_tensor = ss_fea.unsqueeze(0).to(device) if ss_fea is not None else torch.empty(1, 0, seq_len, seq_len, device=device)
    mask = torch.ones(1, 1, seq_len, seq_len, device=device)

    with torch.no_grad():
        logits = model(seq_outer, energy_tensor, ss_fea_tensor, mask).squeeze(0).squeeze(0)
    
    pred_probs = torch.sigmoid(0.5 * (logits + logits.transpose(0, 1))).cpu().numpy()
    os.makedirs(args.final_output_dir, exist_ok=True)
    output_path = os.path.join(args.final_output_dir, f"{sample_name}_rfmfold_prediction.npy")
    np.save(output_path, pred_probs)
    
    print(f"\n✅ Final prediction complete. Probability map saved to: {output_path}")

# ==============================================================================
# 主函数和统一参数解析
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified 2-Stage Inference Pipeline for RFMfold.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 通用和路径参数 ---
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the input FASTA file.')
    parser.add_argument('--ss_feature_dir', type=str, default='./ss_fea', help='Intermediate directory for Stage 1 features.')
    parser.add_argument('--final_output_dir', type=str, default='./final_prediction', help='Directory for the final RFMfold prediction.')
    parser.add_argument('--device', type=str, default='cpu', choices=['gpu', 'cpu'], help='Device for inference.')

    # --- Stage 1 模型参数 ---
    parser.add_argument('--rnaformer_config', type=str, default='./ss_models/ss_models_pth/rnaformer/RNAformer_32M_config_bprna.yml', help='Path to RNAformer config file.')
    parser.add_argument('--rnaformer_state_dict', type=str, default="./ss_models/ss_models_pth/rnaformer/RNAformer_32M_state_dict_bprna.pth", help='Path to RNAformer state_dict.')
    parser.add_argument('--mxfold2_config', type=str, default="./ss_models/ss_models_pth/mxfold2/TR0-canonicals.conf", help='Path to MXfold2 config file.')
    parser.add_argument('--rnafm_state_dict', type=str, default="./ss_models/ss_models_pth/rnafm/RNA-FM-ResNet_bpRNA.pth", help='Path to RNA-FM state_dict.')
    
    # --- Stage 2 模型参数 ---
    parser.add_argument('--rfmfold_checkpoint_path', type=str, default='./rfmfold_ckpt/rfmfold.ckpt', help='Path to the final RFMfold trained checkpoint (.ckpt).')
    parser.add_argument('--energy_dict_path', type=str, default='./bp_fea/avg_energy_stacking_k2.pkl', help='Path to energy stacking dictionary.')
    parser.add_argument('--energy_dist_dict_path', type=str, default='./bp_fea/avg_energy_dist_k2.pkl', help='Path to energy distance dictionary.')

    args = parser.parse_args()
    
    # --- 按顺序执行完整的两阶段流程 ---
    generate_all_ss_features(args)
    run_stage2_rfmfold_inference(args)

if __name__ == "__main__":
    main()