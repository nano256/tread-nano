#!/bin/bash -l
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH -p a100
#SBATCH --time=12:00:00
#SBATCH --job-name=train_mvl


source /export/scratch/ra63vex/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda init
conda activate p310

echo "Number of GPUs allocated: $SLURM_GPUS_ON_NODE"
export MASTER_PORT=$(shuf -i 10000-60000 -n 1)
echo "Selected MASTER_PORT: $MASTER_PORT"

HYDRA_FULL_ERROR=1 srun accelerate launch \
    --mixed_precision bf16 \
    --main_process_port $MASTER_PORT \
    /export/scratch/ra63vex/dev/paper_codebases/tread/train.py \
    log_wandb=True 

