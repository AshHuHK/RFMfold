# data.py

import os
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
import numpy as np

# --- Utility Functions ---

iupac_map: Dict[str, str] = {
    'A':'A', 'C':'C', 'G':'G', 'U':'U', 'T':'U', 'R':'A', 'Y':'C',
    'W':'A', 'S':'G', 'K':'G', 'M':'A', 'B':'C', 'D':'A', 'H':'A',
    'V':'A', 'N':'A', '-':'A'
}
base2idx = {'A':0, 'C':1, 'G':2, 'U':3}

def build_symmetric_energy_matrix(seq: str, energy_dict: Dict, energy_dist_dict: Dict) -> torch.Tensor:
    n = len(seq)
    m = torch.zeros((n, n, 2), dtype=torch.float32)

    def get_kmer_pair(s, i, j, k):
        L = len(s)
        def extract(idx):
            return [s[(idx + d) % L] for d in range(-k, k + 1)]
        return extract(i), extract(j)

    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = get_kmer_pair(seq, i, j, 2)
            key = (''.join(t1), ''.join(t2))
            if key in energy_dict:
                e, o = energy_dict[key], energy_dist_dict[key]
                m[i, j, 0] = m[j, i, 0] = e
                m[i, j, 1] = m[j, i, 1] = o
    return m

def read_fasta(path: Path) -> str:
    return ''.join([
        line.strip().upper() for line in open(path, 'r') if not line.startswith('>')
    ])

def seq_to_onehot(seq: str) -> torch.Tensor:
    onehot = torch.zeros((len(seq), 4), dtype=torch.float32)
    for i, ch in enumerate(seq):
        idx = base2idx[iupac_map.get(ch, 'A')]
        onehot[i, idx] = 1.0
    return onehot

def read_bpseq(path: Path, L: int) -> torch.Tensor:
    mat = torch.zeros((L, L), dtype=torch.float32)
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split()
            i, j = int(parts[0]) - 1, int(parts[2]) - 1
            if j >= 0:
                mat[i, j] = mat[j, i] = 1.0
    return mat

# ============================================================================
#  RNADataset with Automated Feature Scanning
# ============================================================================
class RNADataset(Dataset):
    def __init__(
        self,
        root: str,
        feature_parent_dir: Optional[str] = None,
        energy_dict: Optional[Dict] = None,
        energy_dist_dict: Optional[Dict] = None
    ):
        self.root = Path(root)
        self.fasta_dir = self.root / "fasta"
        self.ss_dir = self.root / "bpseq"
        self.energy_dict = energy_dict
        self.energy_dist_dict = energy_dist_dict

        if not self.root.exists():
            raise FileNotFoundError(f"Root dir {self.root} does not exist")

        fasta_paths = {p.stem.lower(): p for p in self.fasta_dir.glob("*") if p.suffix.lower() in {".fasta", ".fa"}}
        bpseq_paths = {p.stem.lower(): p for p in self.ss_dir.glob("*") if p.suffix.lower() == ".bpseq"}

        # --- Automated feature directory scanning ---
        self.feature_methods = []
        self.feature_maps = {}  # method_name -> {sample_name -> path}
        if feature_parent_dir:
            parent_path = Path(feature_parent_dir)
            if not parent_path.exists():
                raise FileNotFoundError(f"Feature parent dir {parent_path} does not exist")
            
            for method_dir in parent_path.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    print(f"  Scanning feature method: '{method_name}'...")
                    self.feature_methods.append(method_name)
                    npy_map = {p.stem.lower(): p for p in method_dir.rglob("*.npy")}
                    if not npy_map:
                        print(f"  Warning: No .npy files found in {method_dir}")
                    self.feature_maps[method_name] = npy_map
        
        self.feature_methods.sort()

        # --- Intersect samples to find common set ---
        common_samples = set(fasta_paths.keys()) & set(bpseq_paths.keys())
        if not common_samples:
            raise ValueError("No common samples found between fasta and bpseq directories")
        
        for method_name, npy_map in self.feature_maps.items():
            common_samples &= set(npy_map.keys())
            if not common_samples:
                raise ValueError(f"No common samples left after intersecting with feature method '{method_name}'")

        # 【核心修正区域】
        # self.samples 是一个包含最终样本名(字符串)的列表
        self.samples = sorted(list(common_samples))
        
        # 从原始的路径字典中，根据最终的样本名列表来构建映射
        self.fasta_map = {k: fasta_paths[k] for k in self.samples}
        self.bpseq_map = {k: bpseq_paths[k] for k in self.samples}

        #print(f"Found {len(self.samples)} common samples across {len(self.feature_methods)} feature methods.")

        print(f"Found {len(self.samples)} common samples across {len(self.feature_methods)} feature methods.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        key = self.samples[idx]
        seq = read_fasta(self.fasta_map[key])
        L = len(seq)
        
        item = {
            "onehot": seq_to_onehot(seq),
            "adj": read_bpseq(self.bpseq_map[key], L),
            "energy": build_symmetric_energy_matrix(seq, self.energy_dict, self.energy_dist_dict),
            "name": key,
            "length": L,
            "seq": seq,
        }

        # --- Load and process all features for the sample ---
        if self.feature_maps:
            all_ss_feats = []
            for method_name in self.feature_methods:
                if key not in self.feature_maps[method_name]:
                    continue
                
                npy_path = self.feature_maps[method_name][key]
                try:
                    arr = np.load(npy_path)
                    
                    if arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr.squeeze(0)
                    
                    if arr.ndim != 2 or arr.shape[0] != L or arr.shape[1] != L:
                        print(f"  Warning: Skipping feature {npy_path} for sample {key} due to shape mismatch. Got {arr.shape}, expected ({L}, {L}).")
                        continue

                    feat_tensor = torch.from_numpy(arr).float()

                    # Normalize to [0, 1] if values are out of range (logits)
                    if torch.any(feat_tensor > 1.0) or torch.any(feat_tensor < 0.0):
                        feat_tensor = torch.sigmoid(feat_tensor)
                    
                    all_ss_feats.append(feat_tensor.unsqueeze(0))
                except Exception as e:
                    print(f"  Error loading feature {npy_path} for sample {key}: {e}")

            item["ss_prob"] = torch.cat(all_ss_feats, dim=0) if all_ss_feats else None
        
        return item

# ============================================================================
#  Padding and Collation Function
# ============================================================================
def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not batch: return {}
    
    MAX_LEN = max(item["length"] for item in batch)
    B = len(batch)
    
    # --- Initialize padded tensors ---
    seq_batch = torch.zeros((B, MAX_LEN, 4), dtype=torch.float32)
    adj_batch = torch.zeros((B, MAX_LEN, MAX_LEN), dtype=torch.float32)
    ene_batch = torch.zeros((B, MAX_LEN, MAX_LEN, 2), dtype=torch.float32)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    valid_pos = torch.zeros((B, MAX_LEN), dtype=torch.float32)
    
    # --- Handle ss_prob dynamically ---
    has_ss = any("ss_prob" in item and item["ss_prob"] is not None for item in batch)
    ss_batch = None
    if has_ss:
        first_item_with_ss = next(item for item in batch if "ss_prob" in item and item["ss_prob"] is not None)
        C = first_item_with_ss["ss_prob"].shape[0]
        ss_batch = torch.zeros((B, C, MAX_LEN, MAX_LEN), dtype=torch.float32)

    # --- Fill padded tensors ---
    for i, item in enumerate(batch):
        L = item["length"]
        valid_pos[i, :L] = 1.0
        seq_batch[i, :L, :] = item["onehot"]
        adj_batch[i, :L, :L] = item["adj"]
        ene_batch[i, :L, :L, :] = item["energy"]
        if has_ss and "ss_prob" in item and item["ss_prob"] is not None:
            ss_batch[i, :, :L, :L] = item["ss_prob"]
            
    # --- Final transformations ---
    valid_pair = valid_pos.unsqueeze(2) * valid_pos.unsqueeze(1)
    outer = torch.einsum('bif,bjg->bijfg', seq_batch, seq_batch).reshape(B, MAX_LEN, MAX_LEN, 16)
    
    out = {
        "names": [item["name"] for item in batch],
        "seq_str": [item["seq"] for item in batch],
        "seq": seq_batch,
        "adj": adj_batch,
        "seq_outer": outer,
        "energy": ene_batch.permute(0, 3, 1, 2),
        "lengths": lengths,
        "mask": valid_pair,
    }
    if has_ss:
        out["ss_prob"] = ss_batch

    return out