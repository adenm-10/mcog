#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=hsweep
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-15%8

# ===========================================================================
# HORIZON SWEEP. No training, no gradient steps, no GPU.
#
# Why: at horizon 200 the frozen weights cross 19 of 24 doorways with
# probability 1.000. A predictor fitted there is a constant, so p_hat carries
# no signal, A(v,e) has no variance, and handoff-aware cannot differ from
# marginal. Shortening the per-option clock is the one knob that restores
# spread without retraining and without touching the fairness anchor: the
# episode budget (640) is unchanged, and a tighter per-option cap only
# constrains the hierarchy.
#
# Runs tests/probe_edges.py at eight horizons on both fixture sets (16 tasks,
# ~2-5 min each). Output: logs/probe/edges_<tag>_h<H>.json, one per task.
# Then: python -m tests.summarize_horizon_sweep
#
# NO `set -u`. FASRC's /etc/bashrc reads $BASHRCSOURCED with no default, so
# `source ~/.bashrc` under `set -u` kills the shell before anything runs.
# ===========================================================================

set -o pipefail

HORIZONS=(35 40 45 50 55 60 70 90)
RUN_DIRS=(tests/fixtures/regions tests/fixtures_smoke/regions)
TAGS=(n8 n1)

EPISODES=${EPISODES:-30}          # 30 -> SE ~0.09. Rerun the winner at 100.
OUT_DIR=${OUT_DIR:-logs/probe}

NH=${#HORIZONS[@]}
RUN_IDX=$(( SLURM_ARRAY_TASK_ID / NH ))
H_IDX=$((   SLURM_ARRAY_TASK_ID % NH ))

RUN_DIR=${RUN_DIRS[$RUN_IDX]}
TAG=${TAGS[$RUN_IDX]}
H=${HORIZONS[$H_IDX]}
JSON_OUT="${OUT_DIR}/edges_${TAG}_h${H}.json"

cd "${SLURM_SUBMIT_DIR}" || exit 1

# --- fail fast, with a message rather than a traceback ---------------------
[ -f tests/probe_edges.py ] || { echo "FATAL: run sbatch from the repo root"; exit 1; }
[ -f "${RUN_DIR}/resolved_config.yaml" ] || {
  echo "FATAL: no resolved_config.yaml in ${RUN_DIR}"; exit 1; }

mkdir -p "${OUT_DIR}"

# --- provenance BEFORE the env block, so a setup failure still leaves a trace
echo "=== task ${SLURM_ARRAY_TASK_ID}: ${RUN_DIR}  horizon=${H}  episodes=${EPISODES}"
echo "=== out=${JSON_OUT}"
echo "=== host=$(hostname)  date=$(date -Iseconds)"
echo "=== git=$(git rev-parse --short HEAD 2>/dev/null)"

# --- environment (mamba needs an interactive-style shell first) ------------
source ~/.bashrc
module load python
mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"   # numpy GLIBCXX
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1                        # torch is pinned to 1

echo "=== python=$(which python)"

python -m tests.probe_edges \
  --run-dir "${RUN_DIR}" \
  --episodes "${EPISODES}" \
  --horizon "${H}" \
  --gate rect \
  --json-out "${JSON_OUT}"

rc=$?
echo "=== exit ${rc} -> ${JSON_OUT}"
exit ${rc}