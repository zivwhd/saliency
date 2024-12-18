#!/bin/bash

##resource allocation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16g
#SBATCH --gres=gpu:1
#SBATCH --job-name=LTXL
#SBATCH --error=/home/weziv5/work/logs/%x-%j-%4t.err
#SBATCH --output=/home/weziv5/work/logs/%x-%j-%4t.out

hostname
pwd

## Modules

module load "anaconda3/5.3.0"
module load "CUDA/11.8.0"

conda init bash
source activate salsc

echo "Current Env:" $CONDA_DEFAULT_ENV
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

srun python LTX/main/seg_classification/run_seg_cls.py densenet201
echo "DONE"