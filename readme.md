# RFMfold: 

RFMfold is an advanced ensemble learning framework for RNA secondary structure prediction. It uniquely integrates pre-trained RNA foundation models (like RNA-FM), energy parameters, and outputs from other state-of-the-art predictors to achieve enhanced accuracy and flexibility.

This repository provides a ready-to-use inference pipeline as well as a fully customizable training pipeline, allowing users to either get predictions out-of-the-box or build their own specialized ensemble models.

![Model Architecture Diagram](https://raw.githubusercontent.com/Ash-Hu-123/RFMfold/main/pics/overview.png)


## Key Features

- **RNA-Foundation-model**: Default integration with our RNA large language model RNA-FM to provide rich representations and accurate predictions.
- **Ensemble Power**: Leverages a meta-learning approach by combining RNA-FM predictions with base predictions from multiple models.
- **Energy-Aware**: Flexibly incorporates energy parameters as a feature, grounding predictions in biophysical principles.
- **Highly Modular**: Easily integrate or replace base prediction models (e.g. RNAformer, MXfold2) without changing the core architecture.
- **Ready for Inference**: Comes with pre-trained weights for both the base models and the final RFMfold meta-model.
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

## Inference Pipeline

The inference process is fully automated. Given a single FASTA file, the pipeline first generates secondary structure (SS) probability matrices from several base models and then uses them as features for the final RFMfold prediction.

### Quick Start

After installation, you can run inference on a sample FASTA file (`test.fasta`) with a single command.

```bash
# First, activate the conda environment
conda activate RFMfold

# Run the main inference pipeline
python infer_main.py --fasta_file ./test.fasta --device gpu
```

-   `--fasta_file`: Path to your input FASTA file.
-   `--device`: Specify the device to use. Can be `gpu` or `cpu`.

### How It Works

The inference pipeline consists of two stages that run automatically:

1.  **Stage 1: Base Feature Generation**
    -   The script calls our pre-trained base models: **RNA-FM**, **RNAformer**, and **MXfold2**.
    -   Each model predicts a secondary structure probability matrix for the input sequence.
    -   These predictions are saved as `.npy` files inside the `ss_fea/` directory, organized by model name.

2.  **Stage 2: Final RFMfold Prediction**
    -   The main RFMfold model is loaded.
    -   It takes the sequence information, energy parameters, and the **SS features generated in Stage 1** as input.
    -   It produces a final, high-accuracy probability matrix, which is saved in the `final_prediction/` directory.

> **Model Weights**: The default weights for the base models (RNA-FM, RNAformer, MXfold2) are located in the `ss_models/ss_models_pth` directory. If you wish to use different pre-trained models for your specific task, simply replace the corresponding files in this directory.
> 
> **Energy params**: The default energy params for RFMfold are located in the `bp_fea` directory, named as 'avg_energy_stacking_k2.pkl', 'avg_energy_dist_k2.pkl'. If you wish to use different energy params for your specific task, simply replace the corresponding files in this directory.

## Training Your Own Ensemble Model

Beyond direct inference, RFMfold offers exceptional flexibility for creating custom ensembles. You can decide which base prediction methods to integrate and retrain the RFMfold meta-model to specialize in your dataset.

### Preparing Data for Training

To train RFMfold, you need to provide it with pre-computed secondary structure predictions from your chosen base models. Here is a step-by-step guide using the bpRNA dataset as an example.

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

2.  **Generate Probability Matrices**

    For each base model (`method1`, `method2`, etc.), run its prediction on your entire training and validation datasets. Save each output as a 2D probability matrix in `.npy` format. The filename of the `.npy` file must match the name of the corresponding sequence file. For rnafm predictions, run
    ```
    python run_fm_ss.py --input_dir ./my_fasta_files/
    ```
    results will be save in ss_fea/rnafm, change the fm weights if needed.

    
4.  **Configure the Training Script**

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
# Ensure you are in the correct conda environment
conda activate RFMfold

# Run the PyTorch Lightning training script
python pl_train.py
```

The script will automatically detect the feature methods from your directory structure, configure the model channels, and begin training. The model checkpoints with the best validation F1 score will be saved automatically to the `checkpoints_lightning/` directory.
