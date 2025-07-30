# run_rnafm_batch.py


import os
import sys
import torch
import numpy as np
import gc
import collections
from pathlib import Path

sys.path.insert(0, os.path.abspath('./ss_models/RNAFM'))

try:
    import fm
except:
    print(f"❌ Critical Error: Could not load the RNA-FM lib")

def load_rnafm_model(model_path, device):
    """
    Loads the RNA-FM model, alphabet, and batch converter from a state dict file.
    """
    print(f"--- Loading RNA-FM model from: {model_path} ---")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model state dictionary not found at: {model_path}")
    
    # Build the model using the downstream task 'ss' (secondary structure)
    model, alphabet = fm.downstream.build_rnafm_resnet(type="ss", model_location=model_path)
    model.to(device)
    model.eval() # Set the model to evaluation mode
    print("RNA-FM model loaded successfully.")
    return model, alphabet

def run_prediction_for_file(model, batch_converter, device, fasta_path, rnafm_output_dir):
    """
    Reads a single FASTA file, runs RNA-FM prediction, and saves the result.
    """
    try:
        sample_name = fasta_path.stem
        print(f"--- Processing: {sample_name} ---")

        # Read the sequence from the FASTA file
        with open(fasta_path, 'r') as f:
            # Join all lines that don't start with '>', effectively reading the sequence
            seq = "".join([line.strip() for line in f if not line.startswith('>')])

        if not seq:
            print(f"Warning: No sequence found in {fasta_path}. Skipping.")
            return

        # Prepare the data in the format required by the model's batch converter
        data = [(sample_name, seq)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Run inference within a no_grad context for efficiency
        with torch.no_grad():
            results = model({"token": batch_tokens})
            ss_prob_map = results["r-ss"].squeeze(0).cpu().numpy()

        # Define and save the output .npy file
        output_path = rnafm_output_dir / f"{sample_name}.npy"
        np.save(output_path, ss_prob_map)
        print(f"✅ Prediction saved to {output_path}")

    except Exception as e:
        print(f"❌ Failed to process {fasta_path}. Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="""Run RNA-FM predictions for a directory of FASTA files.
This script will find all .fasta or .fa files in the input directory,
run the RNA-FM secondary structure prediction on each, and save the
resulting probability maps as .npy files.""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to the directory containing your input FASTA files.')
    
    parser.add_argument('--output_dir', type=str, default='./ss_fea', 
                        help='Base directory for saving features. The script will create an "rnafm" subdirectory here.')
    
    parser.add_argument('--rnafm_state_dict', type=str, 
                        default="./ss_models/ss_models_pth/rnafm/RNA-FM-ResNet_bpRNA.pth", 
                        help='Path to the RNA-FM state_dict (.pth) file.')
                        
    parser.add_argument('--device', type=str, default='cpu', choices=['gpu', 'cpu'], 
                        help='Device for inference. Select "gpu" to use CUDA if available.')
    
    args = parser.parse_args()
    
    # --- 1. Setup Device and Paths ---
    use_cuda = args.device == 'gpu' and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found at {input_dir}")
        return
    
    # The final output directory will be <base_output_dir>/rnafm/
    rnafm_output_dir = Path(args.output_dir) / 'rnafm'
    rnafm_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved in: {rnafm_output_dir}")

    # --- 2. Load Model ---
    # Load the model only once to be efficient
    try:
        model, alphabet = load_rnafm_model(args.rnafm_state_dict, device)
        batch_converter = alphabet.get_batch_converter()
    except Exception as e:
        print(f"❌ Critical Error: Could not load the RNA-FM model. Please check the path and dependencies.")
        print(f"   Details: {e}")
        return

    # --- 3. Find and Process Files ---
    # Find all files ending with .fasta or .fa
    fasta_files = list(input_dir.glob('*.fasta')) + list(input_dir.glob('*.fa'))
    
    if not fasta_files:
        print(f"Warning: No .fasta or .fa files were found in {input_dir}. Exiting.")
        return
        
    print(f"\nFound {len(fasta_files)} FASTA file(s) to process.")
    
    # Sort the files for consistent processing order
    for fasta_file in sorted(fasta_files):
        run_prediction_for_file(model, batch_converter, device, fasta_file, rnafm_output_dir)

    print("\nScript finished. All files processed.")

if __name__ == "__main__":
    main()