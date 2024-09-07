#!/bin/bash

##resource allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH --job-name=CREATE_ENV

#SBATCH --error=/home/weziv5/work/logs/%x-%j-%4t.err
#SBATCH --output=/home/weziv5/work/logs/%x-%j-%4t.out


module load "anaconda3/5.3.0"
module load "CUDA/11.7.0"

source activate base

# Create a new Conda environment with Python 3.8
CONDA_ENV_NAME=salsc
PYTHON_VERSION=3.8
CUDA_VERSION=11.7

srun nvcc --version


# Create the Conda environment
echo removing old env
conda remove --name salsc --all -y

echo creating env
conda create --name $CONDA_ENV_NAME python=$PYTHON_VERSION -y

echo "Done creating env"
source activate $CONDA_ENV_NAME

echo "Current Env:" $CONDA_DEFAULT_ENV

echo updaing conda
#conda update -n salsc -c defaults conda -y

### echo clean cache
### conda clean --all -y

echo installing ...


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install matplotlib scipy numpy -y

pip3 install scikit-learn
pip3 install scikit-image
pip3 install opencv-python
pip3 install grad-cam
pip3 install pandas


# Verify the installation
echo verifying ...
srun nvcc --version

python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
python -c "import scipy; print('SciPy version:', scipy.__version__)"
python -c "import numpy; print('NumPy version:', numpy.__version__)"
