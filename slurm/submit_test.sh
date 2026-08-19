#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_min_progress_3cm
#SBATCH --output=logs/slurm_staging/%j.out
#SBATCH --error=logs/slurm_staging/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# STAGING SCRATCH SCRIPT -- meant to be overwritten and reused for whatever
# one-off run is next. Full rationale: status.md sec 7.9 (H1) plus this
# session's recalibration of min_progress_cm.
#
# Push, H1 small-domain sparse-HER config (same as push_h1_min_progress_1cm)
# but min_progress_cm raised 1cm -> 3cm. Recalibration this session (real
# closed-loop "re-aim at object" rollouts + HER future-sampling simulation)
# found 3cm passes 98.3% of genuinely competent relabeled goals (vs a naive
# worry that it would be too strict) while cutting the actual trained
# policy's measured HER free-win rate roughly in half again versus 1cm
# (23.7%/35.4% -> 13.2%/17.5% on the two existing checkpoints).
# ===========================================================================

TEMPLATE="push"
TOTAL_STEPS=100000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=false board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_cm=3.0"
RUN_TAG="push_h1_min_progress_3cm"

RUN_DIR="logs/contact/${RUN_TAG}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_TAG=${RUN_TAG}"; echo "TEMPLATE=${TEMPLATE}"; echo "HOST=$(hostname)";
  echo "DATE=$(date -Iseconds)"; echo "TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_contact.py \
  contact=${TEMPLATE} total_steps=${TOTAL_STEPS} seed=0 \
  ${EXTRA_OVERRIDE} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}"
exit $?
