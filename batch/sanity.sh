#!/bin/bash

##resource allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --gres=gpu:1
#SBATCH --job-name=TEST_RUN
#SBATCH --error=/home/weziv5/work/logs/%x-%j.err
#SBATCH --output=/home/weziv5/work/logs/%x-%j.out

## Modules

module load "anaconda3/5.3.0"
module load "CUDA/11.8.0"

conda activate salsc

python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python -c src/dispatch.py