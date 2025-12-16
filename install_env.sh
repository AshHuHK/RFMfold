conda create -n RFMfold python=3.10 -y
CONDA_BASE_PATH=$(conda info --base)
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate RFMfold
conda install nvidia::cuda-nvcc -y
conda install conda-forge::libxcrypt -y
cd ./ss_models/mxfold2
pip install -e .
cd ../RNAformer
pip install -r requirements.txt
pip install ptflops
pip install yacs
cd ..
wget https://archive.org/download/ss_models/ss_models.gz
tar -xvf ss_models.gz
cd ..