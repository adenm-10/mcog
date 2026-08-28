#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=score_v29bc
#SBATCH --output=logs/slurm_staging/%j.out
#SBATCH --error=logs/slurm_staging/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00

# v29 passes (b) and (c). See tools/score_v29_bc.py for the per-arm pins.
OUT="${1:?usage: score_v29_bc.sh <out_dir>}"
mkdir -p "${OUT}" logs/slurm_staging
source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
python tools/score_v29_bc.py --out-dir "${OUT}" --jobs "${SLURM_CPUS_PER_TASK:-4}"
