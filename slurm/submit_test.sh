#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=phaseA_ref
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# PHASE A ACCEPTANCE. Not a sweep. Do not "improve" anything in this file.
#
# Purpose: prove the Phase A refactor moved no numbers, against the only two
# numbers we still have from July (nine_rooms, SAC, seed 1, 2M aggregate):
#     regions  -> composition success_rate = 1.00000
#     monolith -> success_rate             = 0.28125
#     both     -> mean_geodesic_dist       = 10.572200933471322
#
# Every knob is copied verbatim from the July resolved_config, INCLUDING two
# that are known-wrong and are fixed in Phase B with their own commits and
# their own before/after:
#     wall_margin 0.0  (D8/D16: train starts flush to walls, eval keeps 0.25)
#     horizon     200  (submit_sweep.sh's own rule gives 160)
# ===========================================================================

MODES=(regions monolith)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
MAZE=nine_rooms; ALGO=sac; SEED=1; BUDGET=2000000; EVAL_HORIZON=600

if [ "$MODE" = "regions" ]; then
  TRAIN_HORIZON=200; TRAIN_GAMMA=0.995;   LSTART=2000
else
  TRAIN_HORIZON=600; TRAIN_GAMMA=0.99833; LSTART=5000
fi

SWEEP_DIR="logs/phaseA_${SLURM_ARRAY_JOB_ID}"; mkdir -p "${SWEEP_DIR}"
RUN_TAG="dubins_${MODE}_${MAZE}_${ALGO}_s${SEED}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "MODE=${MODE} MAZE=${MAZE} ALGO=${ALGO} SEED=${SEED}";
  echo "TRAIN_HORIZON=${TRAIN_HORIZON} TRAIN_GAMMA=${TRAIN_GAMMA} BUDGET=${BUDGET}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu; export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train.py \
  --algo ${ALGO} --mode ${MODE} --maze-name ${MAZE} \
  --horizon ${TRAIN_HORIZON} --eval-horizon ${EVAL_HORIZON} --gamma ${TRAIN_GAMMA} \
  --arrival-eps 0.4 --omega-max 8.0 --wall-margin 0.0 \
  --total-steps ${BUDGET} --eval-episodes 32 --seed ${SEED} \
  --goal-reward 10 --step-pen 0.01 --collision-pen 0 \
  --switch-gate halfplane \
  --n-envs 8 --gradient-steps 4 --learning-starts ${LSTART} --buffer-size ${BUDGET} \
  --output-dir "${RUN_DIR}/${MODE}_${ALGO}_s${SEED}"
exit $?