import os
import torch
import pytorch_lightning as pl
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# --- 从您的训练脚本中导入必要的模块 ---
# 确保这些文件与此脚本在同一目录下，或者在 Python 路径中
from pl_train import RNASegmenter, RNADataModule, load_from_pickle
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RFMfold model on the validation dataset.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 核心参数: 指定要评估的 checkpoint ---
    parser.add_argument('--ckpt_path', type=str, default="./rfmfold_ckpt/rfmfold.ckpt",
                        help='Path to the .ckpt file you want to evaluate.')
    
    # --- 数据加载参数 (应与训练时保持一致) ---
    parser.add_argument('--val_root', type=str, default="./data/ts0/",
                        help="Path to the root directory of the validation data.")
    parser.add_argument('--feature_parent_dir_val', type=str, default="./ss_fea/",
                        help="Path to the parent directory of validation features.")
    parser.add_argument('--energy_dict_path', type=str, default="./bp_fea/avg_energy_stacking_k2.pkl",
                        help="Path to energy stacking dictionary.")
    parser.add_argument('--energy_dist_dict_path', type=str, default="./bp_fea/avg_energy_dist_k2.pkl",
                        help="Path to energy distance dictionary.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for validation.")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers for the DataLoader.")
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('auto', 'cpu', 'gpu').")
    args = parser.parse_args()

    # --- 1. 检查 Checkpoint 文件是否存在 ---
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found at {args.ckpt_path}")
        return

    # --- 2. 加载模型 ---
    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model_module = RNASegmenter.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=torch.device('cuda' if args.device == 'gpu' else 'cpu') 
    )
    print("Model loaded successfully.")
    
    # --- 3. 设置数据模块 ---
    DATA_CONFIG = {
        "train_root": None, 
        "val_root": args.val_root,
        "energy_dict_path": args.energy_dict_path,
        "energy_dist_dict_path": args.energy_dist_dict_path,
        "feature_parent_dir": {
            "train": None,
            "val": args.feature_parent_dir_val
        }
    }
    LOADER_CONFIG = {
        "train": {}, 
        "val": {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers}
    }
    
    print("\nSetting up validation datamodule...")
    data_module = RNADataModule(DATA_CONFIG, LOADER_CONFIG)
    
    # --- 4. 初始化 Trainer ---
    trainer = pl.Trainer(
        accelerator="gpu" if args.device == 'gpu' else "cpu",
        devices=1, 
        logger=False 
    )
    
    # --- 5. 运行验证 ---
    print("\n🚀 Starting validation... 🚀")
    results = trainer.validate(model=model_module, datamodule=data_module)
    
    # --- 6. 打印结果 ---
    print("\n" + "="*50)
    print("         Validation Results")
    print("="*50)
    if results:
        final_metrics = results[0]
        for key, value in final_metrics.items():
            print(f"{key:<20}: {value:.4f}")
    else:
        print("Validation did not produce any results.")
    print("="*50)


if __name__ == "__main__":
    main()