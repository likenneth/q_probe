#!/bin/bash
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --array=0-39%10

source ~/.bashrc

conda deactivate
conda activate mcts-llm


if [ $(($SLURM_ARRAY_TASK_ID % 2)) == 0 ]; then
  if [ $(($SLURM_ARRAY_TASK_ID % 4)) == 0 ]; then
    python scripts/probe_sweep.py --loss pg --seed $(($SLURM_ARRAY_TASK_ID / 4)) --type 0
  else
    python scripts/probe_sweep.py --loss pg --seed $(($SLURM_ARRAY_TASK_ID / 4)) --type 1
  fi
else
  if [ $(($SLURM_ARRAY_TASK_ID % 4)) == 1 ]; then
    python scripts/probe_sweep.py --loss mse --seed $(($SLURM_ARRAY_TASK_ID / 4)) --type 0
  else
    python scripts/probe_sweep.py --loss mse --seed $(($SLURM_ARRAY_TASK_ID / 4)) --type 1
  fi
fi


if [ $(($SLURM_ARRAY_TASK_ID % 2)) == 0 ]; then
  python scripts/probe_sweep.py --loss ce --seed $(($SLURM_ARRAY_TASK_ID / 2)) --type 0
else
  python scripts/probe_sweep.py --loss ce --seed $(($SLURM_ARRAY_TASK_ID / 2)) --type 1
fi



