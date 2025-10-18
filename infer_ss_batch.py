from ss_gen_batch import load_ss_generating_models, generate_ss_features_for_sequence

import os
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.insert(0, os.path.abspath('.')) # 

from data import read_fasta, seq_to_onehot

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 SS feature generation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- 通用和路径参数 ---
    parser.add_argument('--input_dir', type=str, default='./data/ts0/fasta', help='Path to the directory containing input FASTA files.')
    parser.add_argument('--ss_feature_dir', type=str, default='./ss_fea/', help='Intermediate directory for Stage 1 features.')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference.')

    # --- Stage 1 模型参数 ---
    parser.add_argument('--rnaformer_config', type=str, default='./ss_models/ss_models_pth/rnaformer/RNAformer_32M_config_bprna.yml', help='Path to RNAformer config file.')
    parser.add_argument('--rnaformer_state_dict', type=str, default="./ss_models/ss_models_pth/rnaformer/RNAformer_32M_state_dict_bprna.pth", help='Path to RNAformer state_dict.')
    parser.add_argument('--mxfold2_config', type=str, default="./ss_models/ss_models_pth/mxfold2/TR0-canonicals.conf", help='Path to MXfold2 config file.')
    parser.add_argument('--rnafm_state_dict', type=str, default="./ss_models/ss_models_pth/rnafm/RNA-FM-ResNet_bpRNA.pth", help='Path to RNA-FM state_dict.')
    

    args = parser.parse_args()
    
    # --- SETUP ---
    #device = torch.device("cuda:1" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")
    device = args.device
    input_path = Path(args.input_dir)
    fasta_files = list(input_path.glob('*.fasta')) + list(input_path.glob('*.fa'))
    
    if not fasta_files:
        print(f"Error: No .fasta or .fa files found in directory: {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(fasta_files)} FASTA files to process.")

    # --- STAGE 1: Load all models ONCE ---
    ss_models = load_ss_generating_models(args, device)
    
    for fasta_file in fasta_files:
        print(f"\n{'='*20} Processing: {fasta_file.name} {'='*20}")
        try:
            name = fasta_file.stem
            seq = read_fasta(fasta_file)
            
            # STAGE 1: Generate SS features for the current sequence
            generate_ss_features_for_sequence(seq, name, ss_models, device, args.ss_feature_dir)

        except Exception as e:
            print(f"🚨 Error processing file {fasta_file.name}: {e}")
            print("Skipping to the next file.")
            continue
            
    print("\n🎉 Batch prediction complete for all files. 🎉")


if __name__ == "__main__":
    main()
    
    #/workspace/ash/DAT/BPfold_data/bpRNAnew/fasta