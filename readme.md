# RFMfold: Accurate RNA secondary structure prediction with large RNA language models and ensemble learning

RFMfold is an advanced ensemble learning framework for RNA secondary structure prediction. It uniquely integrates
pre-trained RNA foundation models (like RNA-FM), energy parameters, and outputs from other state-of-the-art predictors
to achieve enhanced accuracy and flexibility.

This repository provides a ready-to-use inference pipeline as well as a fully customizable training pipeline, allowing
users to either get predictions out-of-the-box or build their own specialized ensemble models.

![Model Architecture Diagram](https://github.com/AshHuHK/RFMfold/blob/main/fig.png)


## Table of contents

[//]: # (- [Key Features]&#40;#key-features&#41;)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Inference](#inference)
- [Training](#training)
  - [Data Preparation](#data-preparation)
  - [Start Training](#start-training)
- [Contact](#contact)

[//]: # (## Key Features)

[//]: # ()
[//]: # (- **RNA-Foundation-model**: Default integration with our RNA large language model RNA-FM to provide rich representations)

[//]: # (  and accurate predictions.)

[//]: # (- **Ensemble Power**: Leverages a meta-learning approach by combining RNA-FM predictions with base predictions from)

[//]: # (  multiple models.)

[//]: # (- **Energy-Aware**: Flexibly incorporates energy parameters as a feature, grounding predictions in biophysical)

[//]: # (  principles.)

[//]: # (- **Highly Modular**: Easily integrate or replace base prediction models &#40;e.g. RNAformer, MXfold2&#41; without changing the)

[//]: # (  core architecture.)

[//]: # (- **Trainable**: Provides a complete training pipeline using PyTorch Lightning for users who wish to train RFMfold on)

[//]: # (  their own data or with a custom set of base predictors.)

## Quick Start

Getting RFMfold set up is straightforward. The following steps will create a dedicated `conda` environment with all the
necessary dependencies.

The codes have been tested on a Ubuntu 24.04 system with GeForce RTX 4090 GPU.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/AshHuHK/RFMfold.git
   cd RFMfold
   ```

2. **Run the Installation Script**

   This script will set up a Conda environment named `RFMfold` and install all required packages.

   ```bash
   bash install_env.sh
   conda activate RFMfold
   ```
<<<<<<< HEAD
   
Issue:
fatal error: crypt.h: No such file or directory
conda install conda-forge::libxcrypt

To Do:
1.RNA-FM download link update
=======
>>>>>>> 7d4206449c8e6fc1a7f830cda0e9e170b80c5988

### Inference

The following steps guide you through running RFMfold inference on your own RNA sequences to obtain secondary structure
predictions. We have included example data in the `data/example/` directory for demonstration purposes.

**Step 1. Prepare data**. A `.fasta` file containing a single RNA sequences and a `.bpseq` file containing the native secondary structure is required for each prediction and evaluation.
The `.fasta` and `.bpseq` files should be placed under two separate folders, but ensure that the file names (excluding extensions) match between the `.fasta` and `.bpseq` files.

**Step 2. Generate secondary structure features**. Run the following command to generate secondary structure probability matrices from the base models.

```bash 
python3 infer_ss_batch.py --input_dir ./data/example/fasta --ss_feature_dir ./ss_fea/example
```

**Step 3. Run RFMfold**. Finally, execute the RFMfold prediction script to obtain secondary structure predictions.
```
python3 run_pred.py --val_root ./data/example --feature_parent_dir_val ./ss_fea/example --save_ss_dir ./ss_out/example
```

**Optional Step. Validation**. If `bpseq` files are given under `val_root`, metrics can be computed.

```bash
python3 run_val.py --val_root ./data/example --feature_parent_dir_val ./ss_fea/example
```

On completion, the script prints the **macro/micro F1-score** to the console.

We also offer a instruction for prediction of CASP15 and CASP16 RNA targets.
```bash
   python3 infer_ss_batch.py --input_dir ./data/CASP15/fasta --ss_feature_dir ./ss_fea/CASP15/
   python3 infer_ss_batch.py --input_dir ./data/CASP16/fasta --ss_feature_dir ./ss_fea/CASP16/  
   
   python3 run_val.py --val_root ./data/CASP15/ --feature_parent_dir_val ./ss_fea/CASP15/ 
   python3 run_val.py --val_root ./data/CASP16/ --feature_parent_dir_val ./ss_fea/CASP16/
   
   python3 run_pred.py --val_root ./data/CASP15/ --feature_parent_dir_val ./ss_fea/CASP15/ --save_ss_dir ./ss_out/CASP15/
   python3 run_pred.py --val_root ./data/CASP16/ --feature_parent_dir_val ./ss_fea/CASP16/ --save_ss_dir ./ss_out/CASP16/
   ```

[//]: # (---)

[//]: # ()
[//]: # (## 2&#41; How It Works)

[//]: # ()
[//]: # (### Stage 1 — SS Feature Generation)

[//]: # ()
[//]: # (* Loads pretrained **RNA-FM**, **RNAformer**, **MXfold2**.)

[//]: # (* Predicts per-sequence **SS probability matrices** and saves to:)

[//]: # ()
[//]: # (```text)

[//]: # (ss_fea/)

[//]: # (  rnafm/<name>.npy)

[//]: # (  rnaformer/<name>.npy)

[//]: # (  mxfold2/<name>.npy)

[//]: # (```)

[//]: # ()
[//]: # (### Stage 2 — RFMfold Validation)

[//]: # ()
[//]: # (* Loads the RFMfold model &#40;optionally from a checkpoint&#41;.)

[//]: # (* Inputs: sequence features, **energy params**, and **Stage-1 SS features**.)

[//]: # (* Outputs: **validation F1** &#40;printed/logged&#41;.)

[//]: # ()
[//]: # (---)

[//]: # (Beyond direct inference, RFMfold offers exceptional flexibility for creating custom ensembles. You can decide which base)
[//]: # (prediction methods to integrate and retrain the RFMfold meta-model to specialize in your dataset.)

## Training

### Data Preparation

To train RFMfold, you need to provide it with pre-computed secondary structure predictions from your chosen base models. 
Here is a step-by-step guide using the bpRNA dataset as an example.

1. **Download Data**


2. **Prepare features**

   Inside the `ss_fea/` directory, create `train` and `val` subdirectories. Then, for each base prediction method you
   want to include in your ensemble, create a corresponding subdirectory within both `train` and `val`. For example, the final structure should look like:
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
   
3. **Prepare labels**

   For the training labels (ground truth secondary structures), RFMfold by default use `.bpseq` format and fasta to train the model. The label files should be organized as follows:

    ```
   path/to/dataset/
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

3. **Generate ensembling probability matrices**

   For each base model (`method1`, `method2`, etc.), run its prediction on your entire training and validation datasets.
   Save each output as a 2D probability matrix in `.npy` format. The filename of the `.npy` file must match the name of
   the corresponding sequence file. Run the following two commands to generate the data for the training and validation sets, respectively:
   ```bash
   python3 infer_ss_batch.py --input_dir ./data/bpRNA/TR0/fasta --ss_feature_dir ./ss_fea/bpRNA/TR0
   python3 infer_ss_batch.py --input_dir ./data/bpRNA/VL0/fasta --ss_feature_dir ./ss_fea/bpRNA/VL0
   python3 infer_ss_batch.py --input_dir ./data/bpRNA/TS0/fasta --ss_feature_dir ./ss_fea/bpRNA/TS0
   ```

4. **Set training configuration**
You can customize the training parameters by modifying the configuration file.
A example configuration file `configs/bpRNA.yml` for bpRNA training is provided. You may change the setting for your own needs.

### Start Training

Once your data is prepared and the configuration is set, start the training process:

```bash
# Run the PyTorch Lightning training script
python pl_train.py --config_file configs/bpRNA.yml
```

Afterwards, you may evaluate the trained model on your validation set:

```bash
python3 run_val.py --ckpt_path "./checkpoints_lightning/your_model.ckpt" \
--val_root "./data/bpRNA/TS0/" --feature_parent_dir_val "./ss_fea/bpRNA/TS0/"
```

The script will automatically detect the feature methods from your directory structure, configure the model channels,
and begin training. The model checkpoints with the best validation F1 score will be saved automatically to the
`checkpoints_lightning/` directory.

## Contact

For questions or support, please open an issue on the GitHub repository or contact the authors directly.
