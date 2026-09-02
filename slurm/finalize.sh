#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=finalize
#SBATCH --output=logs/slurm_staging/%j_finalize.out
#SBATCH --error=logs/slurm_staging/%j_finalize.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-03:00:00

# Post-sweep scoring + videos. Submitted automatically by the LAST array task of
# slurm/submit_sweep.sh, so every sweep ends with media on disk and nobody has
# to remember to render.
#
#   sbatch slurm/finalize.sh <sweep_dir> [tag]      # tag defaults to v<jobid>
#
# The benchmark protocol is read from <sweep_dir>/PINS.txt, which the launcher
# wrote. It is not retyped here: a protocol duplicated between a launcher and a
# scorer is how every v25 cross-version comparison came out wrong.
#
# Arms that trained under a LOOSENED task key are still scored here under the
# common tight protocol -- that is the cross-arm comparison. Scoring them a
# second time on their own distribution is a separate, deliberate step.
#
# LOCAL DISK ONLY. Never log these videos to wandb.
set -e

SWEEP="${1:?usage: sbatch slurm/finalize.sh <sweep_dir> [tag]}"
TAG="${2:-$(basename "${SWEEP}" | sed 's/^sweep_/v/')}"
PINS=$(cat "${SWEEP}/PINS.txt")

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu

# Both checkpoints: model_best inverted the arm ordering on v29's gap-assist
# result, so scoring only the final one is not enough to call a verdict.
python tools/score_sweep.py "${SWEEP}" \
  --out-dir "logs/eval/${TAG}" --template push --jobs 4 \
  --ckpt model.zip model_best.zip --pins "${PINS}"

python tools/render_best.py "logs/eval/${TAG}" \
  --pins "${PINS}" --media-dir "media/${TAG}" --n 6

echo "done: logs/eval/${TAG} and media/${TAG}"
