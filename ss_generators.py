# ss_generators.py

import os
import sys
import torch
import numpy as np
import gc
import collections
from pathlib import Path

sys.path.insert(0, os.path.abspath('./ss_models/RNAformer'))
sys.path.insert(0, os.path.abspath('./ss_models/mxfold2'))
sys.path.insert(0, os.path.abspath('./ss_models/RNAFM'))

try:
    from RNAformer.model.RNAformer import RiboFormer
    from RNAformer.utils.configuration import Config
    import loralib as lora
    from mxfold2.fold.mix import MixedFold
    import mxfold2.param_turner2004 as param_turner2004
    from mxfold2.fold.rnafold import RNAFold
    from mxfold2.fold.zuker import ZukerFold
    import fm
except ImportError as e:
    print(f"Error: Could not import a required submodule for SS generation. Details: {e}")
    sys.exit(1)

def insert_lora_layer(model, ft_config):
    lora_config = { "r": ft_config.r, "lora_alpha": ft_config.lora_alpha, "lora_dropout": ft_config.lora_dropout }
    with torch.no_grad():
        for name, module in model.named_modules():
            if any(replace_key in name for replace_key in ft_config.replace_layer):
                parent = model.get_submodule(".".join(name.split(".")[:-1])); target_name = name.split(".")[-1]; target = model.get_submodule(name)
                if isinstance(target, torch.nn.Linear) and "qkv" in name:
                    new_module = lora.MergedLinear(target.in_features, target.out_features, bias=target.bias is not None, enable_lora=[True, True, True], **lora_config); new_module.weight.copy_(target.weight)
                    if target.bias is not None: new_module.bias.copy_(target.bias)
                elif isinstance(target, torch.nn.Linear):
                    new_module = lora.Linear(target.in_features, target.out_features, bias=target.bias is not None, **lora_config); new_module.weight.copy_(target.weight)
                    if target.bias is not None: new_module.bias.copy_(target.bias)
                elif isinstance(target, torch.nn.Conv2d):
                    kernel_size = target.kernel_size[0]; new_module = lora.Conv2d(target.in_channels, target.out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=target.bias is not None, **lora_config); new_module.conv.weight.copy_(target.weight)
                    if target.bias is not None: new_module.conv.bias.copy_(target.bias)
                setattr(parent, target_name, new_module)
    return model
def sequence2index_vector(sequence, mapping):
    int_sequence = [mapping.get(char.upper(), mapping['N']) for char in sequence]
    return torch.LongTensor(int_sequence)
def create_contact_map_mxfold2(bps, seq_len):
    contact_map = np.zeros((seq_len, seq_len), dtype=np.int8)
    for i in range(1, seq_len + 1):
        j = bps[i]
        if j > 0: contact_map[i - 1, j - 1] = 1
    return contact_map
def load_config_from_file_mxfold2(filepath: str) -> dict:
    def convert_type(value_str):
        try: return int(value_str)
        except ValueError:
            try: return float(value_str)
            except ValueError: return value_str
    TUPLE_KEYS = {'num_filters','filter_size','pool_size','num_hidden_units','num_paired_filters','paired_filter_size'}
    file_args = collections.defaultdict(list)
    with open(filepath, 'r') as f: lines = [line.strip() for line in f if line.strip()]
    for key_str, value_str in zip(lines[0::2], lines[1::2]):
        if not key_str.startswith('--'): continue
        key = key_str.lstrip('-').replace('-', '_'); file_args[key].append(convert_type(value_str))
    final_config = {}
    for key, values in file_args.items():
        if key in TUPLE_KEYS or len(values) > 1: final_config[key] = tuple(values)
        else: final_config[key] = values[0]
    return final_config
def build_model_mxfold2(config):
    model_name = config.get('model', 'Turner')
    if model_name == 'Zuker': model = ZukerFold(model_type='M', **config)
    elif model_name == 'Mix': model = MixedFold(init_param=param_turner2004, **config)
    elif model_name == 'Turner': model = RNAFold(param_turner2004)
    elif model_name == 'MixC': model = MixedFold(init_param=param_turner2004, model_type='C', **config)
    else: raise ValueError(f"Model type '{model_name}' not implemented")
    return model
def release_memory(model_name: str, device: str):
    print(f"\n--- Releasing memory after {model_name} ---")
    gc.collect()
    if device == 'gpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

# --- 各个模型的私有运行函数 ---
def _run_rnaformer(args, device):
    print("\n--- Running RNAformer Prediction ---")
    config = Config(config_file=args.rnaformer_config)
    model = RiboFormer(config.RNAformer)
    state_dict = torch.load(args.rnaformer_state_dict, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    save_dir = os.path.join(args.ss_feature_dir, 'rnaformer')
    os.makedirs(save_dir, exist_ok=True)
    seq_vocab = ['A', 'C', 'G', 'U', 'N']
    seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))
    name_x = Path(args.fasta_file).stem
    with open(args.fasta_file, 'r') as file: seq_x = "".join([l.strip() for l in file if not l.startswith('>')])
    sequence_tensor = sequence2index_vector(seq_x, seq_stoi).unsqueeze(0).to(device)
    src_len = torch.LongTensor([sequence_tensor.shape[-1]]).to(device)
    pdb_sample_tensor = torch.FloatTensor([[1]]).to(device)
    with torch.no_grad():
        logits, _ = model(sequence_tensor, src_len, pdb_sample_tensor)
        pred_mat = (torch.sigmoid(logits[0, :, :, -1].to(torch.float32))).cpu().numpy()
    output_path = os.path.join(save_dir, f"{name_x}.npy")
    np.save(output_path, pred_mat)
    print(f"RNAformer prediction saved to {output_path}")

def _run_mxfold2(args, device):
    print("\n--- Running MXfold2 Prediction ---")
    config = load_config_from_file_mxfold2(args.mxfold2_config)
    model = build_model_mxfold2(config)
    param_path = os.path.join(os.path.dirname(args.mxfold2_config), config.get('param'))
    state_dict = torch.load(param_path, map_location=device)
    if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    save_dir = os.path.join(args.ss_feature_dir, 'mxfold2')
    os.makedirs(save_dir, exist_ok=True)
    with open(args.fasta_file, 'r') as f: seq = "".join([l.strip() for l in f if not l.startswith('>')])
    with torch.no_grad():
        _, _, bps = model([seq])
        contact_map = create_contact_map_mxfold2(bps[0], len(seq))
    name_x = Path(args.fasta_file).stem
    output_path = os.path.join(save_dir, f"{name_x}.npy")
    np.save(output_path, contact_map.astype(np.float16))
    print(f"MXfold2 prediction saved to {output_path}")

def _run_rnafm(args, device):
    print("\n--- Running RNA-FM Prediction ---")
    model, alphabet = fm.downstream.build_rnafm_resnet(type="ss", model_location=args.rnafm_state_dict)
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()
    save_dir = os.path.join(args.ss_feature_dir, 'rnafm')
    os.makedirs(save_dir, exist_ok=True)
    name_x = Path(args.fasta_file).stem
    with open(args.fasta_file, 'r') as f: seq = "".join([l.strip() for l in f if not l.startswith('>')])
    data = [(name_x, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model({"token": batch_tokens})
        ss_prob_map = results["r-ss"].squeeze(0).cpu().numpy()
    output_path = os.path.join(save_dir, f"{name_x}.npy")
    np.save(output_path, ss_prob_map)
    print(f"RNA-FM prediction saved to {output_path}")

# --- 公开的接口函数 ---
def generate_all_ss_features(args):
    
    print("==============================================")
    print("= STAGE 1: Generating Secondary Structure Features =")
    print("==============================================")

    device = torch.device("cuda" if args.device == 'gpu' and torch.cuda.is_available() else "cpu")

    _run_rnaformer(args, device)
    release_memory("RNAformer", args.device)
    
    _run_mxfold2(args, device)
    release_memory("MXfold2", args.device)
    
    _run_rnafm(args, device)
    release_memory("RNA-FM", args.device)

    print("\n--- Stage 1 complete. All SS features generated. ---")