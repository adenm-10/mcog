#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=dubins_sweep
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# 16 = 2 modes x 2 mazes x 2 algos x 2 seeds.  (sac_her and 'large' dropped.)
# index decode (slowest->fastest): mode, maze, algo, seed
MODES=(monolith regions); MAZES=(nine_rooms giant)
ALGOS=(ppo sac);          SEEDS=(1 2)
i=$SLURM_ARRAY_TASK_ID
MODE=${MODES[$((  i / 8 ))]}
MAZE=${MAZES[$(( (i % 8) / 4 ))]}
ALGO=${ALGOS[$(( (i % 4) / 2 ))]}
SEED=${SEEDS[$((  i % 2 ))]}

# ---------------------------------------------------------------------------
# Per-maze task knobs. All chosen so NOTHING is horizon/step constrained:
#   HORIZON      = ceil(1.75 x geodesic_diameter_steps)   (diam: nine=320, giant=310)
#   GAMMA        set so 1/(1-gamma) = HORIZON  -> gamma^H = e^-1 = 0.37 (terminal bonus visible)
#   BUDGET       = 2.0M env steps (both mazes; past the plateau seen at 1.5-1.8M)
#   H_REGION     = ceil(2.0 x max_region_diameter_steps)  (nine=80, giant=140 -> old 130 was BINDING)
#   GAMMA_REGION matched to H_REGION the same way.
# EVAL_HORIZON is ALWAYS the full-maze HORIZON (monolith-eval == composition-eval head-to-head).
case $MAZE in
  nine_rooms) HORIZON=600; GAMMA=0.99833; BUDGET=2000000; H_REGION=200; GAMMA_REGION=0.995   ;;
  giant)      HORIZON=550; GAMMA=0.99818; BUDGET=2000000; H_REGION=300; GAMMA_REGION=0.99667 ;;
  *) echo "unknown maze '$MAZE'"; exit 2 ;;
esac

# Horizon/gamma branch by mode: monolith at full-maze scale, regions at one-room scale.
if [ "$MODE" = "regions" ]; then
  TRAIN_HORIZON=$H_REGION; TRAIN_GAMMA=$GAMMA_REGION
else
  TRAIN_HORIZON=$HORIZON;  TRAIN_GAMMA=$GAMMA
fi
EVAL_HORIZON=$HORIZON

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_TAG="dubins_${MODE}_${MAZE}_${ALGO}_s${SEED}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "MODE=${MODE} MAZE=${MAZE} ALGO=${ALGO} SEED=${SEED} TRAIN_HORIZON=${TRAIN_HORIZON} EVAL_HORIZON=${EVAL_HORIZON} TRAIN_GAMMA=${TRAIN_GAMMA} BUDGET=${BUDGET}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# JAX (Dubins dynamics) pinned to CPU -> Subproc-safe. torch/SB3 nets keep the A100.
export JAX_PLATFORMS=cpu; export CUDA_VISIBLE_DEVICES=0; export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Task/eval knobs identical across all arms; reward = SPARSE for all (goal bonus + step pen).
COMMON="--algo ${ALGO} --mode ${MODE} --maze-name ${MAZE} \
        --horizon ${TRAIN_HORIZON} --eval-horizon ${EVAL_HORIZON} --gamma ${TRAIN_GAMMA} \
        --arrival-eps 0.4 --omega-max 8.0 \
        --total-steps ${BUDGET} --eval-episodes 32 --seed ${SEED} \
        --goal-reward 10 --step-pen 0.01 --collision-pen 0 \
        --output-dir ${RUN_DIR}/${MODE}_${ALGO}_s${SEED}"

if [ "$ALGO" = "ppo" ]; then
  # On-policy. n_steps = TOTAL rollout across envs (loader divides by n_envs internally).
  # monolith: 16384 -> ~122 updates @ 2.0M.  regions: 4096 -> ~55-60 updates/region
  # (4096//32=128, clears the max(128,..) floor exactly; fixes the 12-update starvation).
  if [ "$MODE" = "regions" ]; then N_STEPS=4096; else N_STEPS=16384; fi
  python train.py $COMMON \
    --n-envs 32 --ent-coef 0.01 --n-steps ${N_STEPS}
else
  # SAC. gradient_steps=4 both modes -> equal UTD. buffer=BUDGET -> full-retention replay.
  if [ "$MODE" = "regions" ]; then LSTART=2000; else LSTART=5000; fi
  python train.py $COMMON \
    --n-envs 8 --gradient-steps 4 --learning-starts ${LSTART} --buffer-size ${BUDGET}
fi

rc=$?
still=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}" -t PENDING,RUNNING -o "%A_%a" \
        | grep -v "^${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}$" | wc -l)
if [ "${still}" -eq 0 ]; then
  mkdir -p "${SWEEP_DIR}/slurm_logs"
  find logs/slurm_staging -maxdepth 1 -type f \
       \( -name "${SLURM_ARRAY_JOB_ID}_*.out" -o -name "${SLURM_ARRAY_JOB_ID}_*.err" \) \
       -exec mv -t "${SWEEP_DIR}/slurm_logs/" {} + 2>/dev/null
fi
exit $rc