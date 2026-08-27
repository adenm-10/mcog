#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=score_sweep
#SBATCH --output=logs/slurm_staging/%j.out
#SBATCH --error=logs/slurm_staging/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=0-01:00:00

# Score a whole contact sweep with eval_contact.py. Each eval is ~13s of CPU
# and 60 episodes, so 36 of them belong on a compute node, not the login node
# (nproc=1 there, and it is already at load ~40).
#
#   sbatch slurm/score_sweep.sh logs/sweep_42056320 logs/eval/v26_42056320 base
#
# $1 sweep dir   $2 out dir   $3 arm to also score under every other digest group

SWEEP="${1:?usage: score_sweep.sh <sweep_dir> <out_dir> [transfer_arm]}"
OUT="${2:?usage: score_sweep.sh <sweep_dir> <out_dir> [transfer_arm]}"
TRANSFER="${3:-}"

mkdir -p "${OUT}" logs/slurm_staging

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

ARGS=(--out-dir "${OUT}" --jobs "${SLURM_CPUS_PER_TASK:-4}")
[ -n "${TRANSFER}" ] && ARGS+=(--transfer-arm "${TRANSFER}")

python tools/score_sweep.py "${SWEEP}" "${ARGS[@]}"
