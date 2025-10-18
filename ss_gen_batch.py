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

def _run_rnaformer(seq, name_x, model, device, save_dir):
    """(Private) Runs inference for a single sequence using a pre-loaded RNAformer model."""
    seq_vocab = ['A', 'C', 'G', 'U', 'N']
    seq_stoi = dict(zip(seq_vocab, range(len(seq_vocab))))
    sequence_tensor = sequence2index_vector(seq, seq_stoi).unsqueeze(0).to(device)
    src_len = torch.LongTensor([sequence_tensor.shape[-1]]).to(device)
    pdb_sample_tensor = torch.FloatTensor([[1]]).to(device)
    with torch.no_grad():
        logits, _ = model(sequence_tensor, src_len, pdb_sample_tensor)
        pred_mat = (torch.sigmoid(logits[0, :, :, -1].to(torch.float32))).cpu().numpy()
    output_path = os.path.join(save_dir, 'rnaformer', f"{name_x}.npy")
    np.save(output_path, pred_mat)

def _run_mxfold2(seq, name_x, model, device, save_dir):
    """(Private) Runs inference for a single sequence using a pre-loaded MXfold2 model."""
    with torch.no_grad():
        _, _, bps = model([seq])
        contact_map = create_contact_map_mxfold2(bps[0], len(seq))
    output_path = os.path.join(save_dir, 'mxfold2', f"{name_x}.npy")
    np.save(output_path, contact_map.astype(np.float16))

def _run_rnafm(seq, name_x, model, alphabet, device, save_dir):
    """(Private) Runs inference for a single sequence using a pre-loaded RNA-FM model."""
    batch_converter = alphabet.get_batch_converter()
    data = [(name_x, seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model({"token": batch_tokens})
        ss_prob_map = torch.sigmoid(results["r-ss"]).squeeze(0).cpu().numpy()
    output_path = os.path.join(save_dir, 'rnafm', f"{name_x}.npy")
    np.save(output_path, ss_prob_map)


# --- NEW: 分离模型加载和推理 ---

def load_ss_generating_models(args, device):
    """
    (NEW) Loads all Stage 1 models into memory once.
    Returns a dictionary of loaded models.
    """
    print("\n--- Loading all Stage 1 models... ---")
    models = {}

    # Load RNAformer
    print("Loading RNAformer...")
    config = Config(config_file=args.rnaformer_config)
    rnaformer_model = RiboFormer(config.RNAformer)
    state_dict = torch.load(args.rnaformer_state_dict, map_location=device)
    rnaformer_model.load_state_dict(state_dict, strict=True)
    rnaformer_model.to(device).eval()
    models['rnaformer'] = rnaformer_model
    
    # Load MXfold2
    print("Loading MXfold2...")
    config_mxfold = load_config_from_file_mxfold2(args.mxfold2_config)
    mxfold2_model = build_model_mxfold2(config_mxfold)
    param_path = os.path.join(os.path.dirname(args.mxfold2_config), config_mxfold.get('param'))
    state_dict_mxfold = torch.load(param_path, map_location=device)
    if 'model_state_dict' in state_dict_mxfold: state_dict_mxfold = state_dict_mxfold['model_state_dict']
    mxfold2_model.load_state_dict(state_dict_mxfold)
    mxfold2_model.to(device).eval()
    models['mxfold2'] = mxfold2_model

    # Load RNA-FM
    print("Loading RNA-FM...")
    rnafm_model, alphabet = fm.downstream.build_rnafm_resnet(type="ss", model_location=args.rnafm_state_dict)
    rnafm_model.to(device).eval()
    models['rnafm'] = (rnafm_model, alphabet) # RNA-FM needs alphabet too

    print("All Stage 1 models loaded successfully.")
    return models

def generate_ss_features_for_sequence(seq, name, loaded_models, device, save_dir):
    """
    (NEW) Generates all SS features for a single sequence using pre-loaded models.
    """
    print(f"\n--- Generating SS features for: {name} ---")
    
    # 确保每个子目录都存在
    for model_name in loaded_models.keys():
        os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)
        
    _run_rnaformer(seq, name, loaded_models['rnaformer'], device, save_dir)
    print(f"RNAformer prediction saved for {name}")

    _run_mxfold2(seq, name, loaded_models['mxfold2'], device, save_dir)
    print(f"MXfold2 prediction saved for {name}")
    
    rnafm_model, alphabet = loaded_models['rnafm']
    _run_rnafm(seq, name, rnafm_model, alphabet, device, save_dir)
    print(f"RNA-FM prediction saved for {name}")