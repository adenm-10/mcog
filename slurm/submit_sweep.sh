#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=obshygiene_objframe_herfix
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# Combined integration sweep (env hygiene + recontact object-frame goal +
# push HER-relabel fix), all now baked into the code rather than sweepable
# flags -- gate-verified (4/4) plus synthetic checks (obs shape/scale,
# inactive-finger masking, object-frame invariance, HER-relabel recompute,
# disturbance-gated compute_reward) and end-to-end smoke runs before this
# sweep was written.
#
# What changed vs. the previous (restrict_contact_actions / dropfingervel)
# sweep, and why it applies to both templates:
#  - physics.py's obs() drops no_contact_steps/peak_force (guard-only,
#    unused by any obs() consumer) and normalizes positions/velocities --
#    was an unbounded tick counter sitting next to ~[-1,1] features.
#  - push's inactive finger is now masked in ContactEnv.step() -- it had no
#    task-relevant signal, and its own exploration noise could trigger
#    forbidden_contact independent of the active finger's behavior.
#  - recontact's goal is now object-frame (gym_env.py's _achieved_xy), not a
#    world point fixed at reset -- stays valid as a HER-relabel target even
#    if the object moves between the two ticks a relabel pairs together.
#  - recontact also gates on a persistent "object was ever disturbed this
#    episode" flag, not just settled-at-arrival -- closes the "bulldoze then
#    let it settle right before arriving" loophole.
#  - push gets a custom HerReplayBuffer (her_buffer.py) that recomputes the
#    now-stale target-relative observation slot on every relabeled
#    transition, instead of leaving the original episode's target baked in
#    (confirmed via SB3 source: relabeling only ever swaps desired_goal).
#
# restrict_contact_actions is deliberately left at its default (false) and
# not swept here -- this sweep tests whether the fixes above help the
# general, UNRESTRICTED action case, independent of that still-open idea.
#
# Tasks 0-1: push, 2 seeds, unrestricted actions. Two changes from the prior
# sweep's push cell: guard_terminates false->true (episode now ends at
# contact_lost instead of running ~190 dead ticks that flooded HER's
# future-goal sampling with near-duplicate resting-position targets), and
# min_progress_cm 3.0->0.5 (was killing HER credit for any real 1-3cm
# progress -- since it's the ONLY channel carrying signal here, given
# reward is pure sparse and push almost never reaches a real full-distance
# settled success yet). 0.5 matches arrival_eps (0.4), clearing the
# deterministic 0.00cm no-op floor without suppressing small real progress.
#
# Tasks 2-3: push, 2 seeds, same as tasks 0-1 plus require_settled=false --
# push's real success (push_arrival's reached_interface) normally requires
# the object to be both close AND settled (low velocity), while HER's
# compute_reward was already position-only for push; this closes that
# train/eval gap from the other direction (loosen real to match virtual,
# instead of tightening virtual to match real, which would make an
# already-too-sparse signal sparser).
#
# Tasks 4-7 (recontact, guard_terminates true/false x 2 seeds) are already
# running fine under job 40944664 and are UNCHANGED by the fixes below --
# array is 0-3 this round, push only, so they aren't duplicated. Their
# branch is left in place below for reuse if this script is resubmitted
# later without editing it back in.
#
# What changed vs. job 40944664's push cells (code only, no new overrides --
# EXTRA_OVERRIDE below is identical to that run's):
#  - her_buffer.py's _get_virtual_samples now marks a relabeled transition
#    "done" whenever it scores the arrival bonus, instead of keeping the
#    original rollout's done flag. Previously the critic kept bootstrapping
#    (r + gamma*Q(next)) past a virtual arrival instead of stopping there --
#    a likely driver of the critic_loss blowup seen once min_progress_cm's
#    3.0->0.5 drop (job 40944664) made virtual arrivals common.
#  - min_progress_cm's check is now PER-PAIR, not per-episode: it compares
#    the relabeled goal against the object's position right before the
#    transition being relabeled (gym_env.py's new `pre_achieved_goal`),
#    not the episode's start position. A goal set to wherever the object
#    already was sitting is a free win regardless of how far the episode
#    moved elsewhere, and a goal reached only after real movement is
#    informative regardless of the episode's overall displacement -- the
#    old episode-start comparison couldn't tell these apart. Still rejects
#    the original degenerate case (a fully static episode has pre==goal
#    everywhere too), so nothing is reopened.
# ===========================================================================

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 4 ]; then
  TEMPLATE="push"
  SEED=$(( i % 2 ))
  TOTAL_STEPS=150000
  EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_cm=0.5 her_n_sampled_goal=4 learning_starts=10000 same_room_goal_prob=0.0"
  if [ "$i" -lt 2 ]; then
    RUN_TAG="push_objhygiene_s${SEED}"
  else
    EXTRA_OVERRIDE="${EXTRA_OVERRIDE} require_settled=false"
    RUN_TAG="push_nosettled_s${SEED}"
  fi
else
  TEMPLATE="recontact"
  GT=$([ "$i" -lt 6 ] && echo "true" || echo "false")
  SEED=$(( (i - 4) % 2 ))
  TOTAL_STEPS=1000000
  EXTRA_OVERRIDE="use_her=true w_T=0.0 w_a=0.0 w_m=0.0 guard_terminates=${GT}"
  RUN_TAG="recontact_objframe_gt${GT}_s${SEED}"
fi

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)"; } > "${RUN_DIR}/meta.txt"
cp "$0" "${RUN_DIR}/submit_script.sh"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python train_contact.py \
  contact=${TEMPLATE} total_steps=${TOTAL_STEPS} seed=${SEED} \
  ${EXTRA_OVERRIDE} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="obshygiene_objframe_herfix_${SLURM_ARRAY_JOB_ID}"
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
