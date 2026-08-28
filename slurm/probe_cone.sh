#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=probe_cone
#SBATCH --output=logs/slurm_staging/%j.out
#SBATCH --error=logs/slurm_staging/%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem=24G --time=0-01:00:00
mkdir -p logs/slurm_staging
source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
python tools/probe_goal_diversity.py --jobs "${SLURM_CPUS_PER_TASK:-4}"
