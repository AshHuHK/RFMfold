# RFMfold: Accurate RNA secondary structure prediction with large RNA language models and ensemble learning

RFMfold is an advanced ensemble learning framework for RNA secondary structure prediction. It uniquely integrates pre-trained RNA foundation models (like RNA-FM), energy parameters, and outputs from other state-of-the-art predictors to achieve enhanced accuracy and flexibility.

This repository provides a ready-to-use inference pipeline as well as a fully customizable training pipeline, allowing users to either get predictions out-of-the-box or build their own specialized ensemble models.

![Model Architecture Diagram](https://github.com/AshHuHK/RFMfold/blob/main/fig.png)


## Key Features

- **RNA-Foundation-model**: Default integration with our RNA large language model RNA-FM to provide rich representations and accurate predictions.
- **Ensemble Power**: Leverages a meta-learning approach by combining RNA-FM predictions with base predictions from multiple models.
- **Energy-Aware**: Flexibly incorporates energy parameters as a feature, grounding predictions in biophysical principles.
- **Highly Modular**: Easily integrate or replace base prediction models (e.g. RNAformer, MXfold2) without changing the core architecture.
- **Trainable**: Provides a complete training pipeline using PyTorch Lightning for users who wish to train RFMfold on their own data or with a custom set of base predictors.

## Installation

Getting RFMfold set up is straightforward. The following steps will create a dedicated `conda` environment with all the necessary dependencies.

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Ash-Hu-123/RFMfold.git
    cd RFMfold
    ```

2.  **Run the Installation Script**

    This script will set up a Conda environment named `RFMfold` and install all required packages.

    ```bash
    bash install_env.sh
    ```    > **Note**: During the installation process, you may be prompted to confirm installations. Please answer `yes` to all prompts to ensure a complete setup.

# RFMfold — Validation Pipeline

Minimal, logical steps to reproduce validation F1 on your dataset.

## 1) Steps

Prepare your validation data, the final architecture should be as follows, make sure **basenames match** across FASTA / BPSEQ (e.g., `foo.fasta` ↔ `foo.bpseq`).

### Step A — Prepare data  
Place your files as:  
```
./data/ts0/fasta/*.fasta  
./data/ts0/bpseq/*.bpseq
```
the file architecture should be as follows,
```text
project/
├─ data/ts0/
│  ├─ fasta/        # input .fasta
│  └─ bpseq/        # reference .bpseq
```
### Step B — Generate SS features (Stage 1)
```bash 
python3 infer_ss_batch.py \
  --input_dir ./data/ts0/fasta \
  --ss_feature_dir ./ss_fea
```

### Step C — Run RFMfold validation (Stage 2)
```
#sets: val_root = ./data/ts0/ in run_val.py
python3 run_val.py
```

On completion, the script prints **macro/micro F1** to the console.

---

## 2) How It Works

### Stage 1 — SS Feature Generation

* Loads pretrained **RNA-FM**, **RNAformer**, **MXfold2**.
* Predicts per-sequence **SS probability matrices** and saves to:

```text
ss_fea/
  rnafm/<name>.npy
  rnaformer/<name>.npy
  mxfold2/<name>.npy
```

### Stage 2 — RFMfold Validation

* Loads the RFMfold model (optionally from a checkpoint).
* Inputs: sequence features, **energy params**, and **Stage-1 SS features**.
* Outputs: **validation F1** (printed/logged).

---

Beyond direct inference, RFMfold offers exceptional flexibility for creating custom ensembles. You can decide which base prediction methods to integrate and retrain the RFMfold meta-model to specialize in your dataset.

# RFMfold — Training Pipeline

### Preparing Data for Training

To train RFMfold, you need to provide it with pre-computed secondary structure predictions from your chosen base models for example running `infer_ss_batch.py`. Here is a step-by-step guide using the bpRNA dataset as an example.

1.  **Create the Directory Structure**

    Inside the `ss_fea/` directory, create `train` and `val` subdirectories. Then, for each base prediction method you want to include in your ensemble, create a corresponding subdirectory within both `train` and `val`.

    The final structure should look like this:
    ```
    ss_fea/
    ├── train/
    │   ├── method1/
    │   │   ├── sequence1.npy
    │   │   ├── sequence2.npy
    │   │   └── ...
    │   ├── method2/
    │   │   ├── sequence1.npy
    │   │   └── ...
    │   └── ...
    └── val/
        ├── method1/
        │   ├── sequence_val_1.npy
        │   └── ...
        ├── method2/
        │   ├── sequence_val_1.npy
        │   └── ...
        └── ...
     ```
2.  **Create the Training Data**
   
       For the training labels and data, RFMfold by default use bpseq format and fasta to train, by default the data should organized to look like below,
   
     ```
    xx_dataset/
    ├── train/
    │   ├── bpseq/
    │   │   ├── sequence1.bpseq
    │   │   ├── sequence2.bpseq
    │   │   └── ... 
    │   │
    │   └── fasta/
    │       ├── sequence1.fasta
    │       ├── sequence2.fasta
    │       └── ...
    │
    └── val/
        ├── bpseq/
        │   ├── val_sequence1.bpseq
        │   ├── val_sequence2.bpseq
        │   └── ...
        │
        └── fasta/
            ├── val_sequence1.fasta
            ├── val_sequence2.fasta
            └── ...
          
4.  **Generate Ensembling Probability Matrices**

    For each base model (`method1`, `method2`, etc.), run its prediction on your entire training and validation datasets. Save each output as a 2D probability matrix in `.npy` format. The filename of the `.npy` file must match the name of the corresponding sequence file. For example run
    ```
    python3 infer_ss_batch.py --input_dir ./data/val/fasta --ss_feature_dir ./ss_fea/val
    ```
    results will be save in ss_fea/val.

    
5.  **Configure the Training Script**

    Open the `pl_train.py` script and locate the `DATA_CONFIG` dictionary. Update the directory paths to point to your datasets and feature locations.

    ```python
    # Inside pl_train.py
    DATA_CONFIG = {
        "train_root": "/path/to/your/bprna/TR0", # Contains fasta/ and bpseq/ for training
        "val_root": "/path/to/your/bprna/TS0",   # Contains fasta/ and bpseq/ for validation
        "energy_dict_path": "./bp_fea/avg_energy_stacking_k2.pkl",
        "energy_dist_dict_path": "./bp_fea/avg_energy_dist_k2.pkl",
        "feature_parent_dir": {
            "train": "./ss_fea/train", # Points to your generated train features
            "val": "./ss_fea/val"      # Points to your generated validation features
        }
    }
    ```

### Start Training

Once your data is prepared and the configuration is set, start the training process:

```bash
# Run the PyTorch Lightning training script
python pl_train.py
```

The script will automatically detect the feature methods from your directory structure, configure the model channels, and begin training. The model checkpoints with the best validation F1 score will be saved automatically to the `checkpoints_lightning/` directory.
