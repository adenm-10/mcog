#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_contact
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-23
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v29 PUSH sweep: which of the scaffolds is load-bearing?
#
# 24 cells = arm {nogapassist, unmask, rawact, randtheta, physdamp, hardmode,
#                 narrowgap, bigroom} x seed {0,1,2}, 400k steps each.
#
# (The previous contents were job 42248679's launcher. All 18 of its per-run
# copies, logs/sweep_42248679/*/submit_script.sh, are byte-identical to the
# copy overwritten here -- verified before overwriting.)
#
# WHY. v28 got push to 0.739 on the same-room benchmark (from 0.21), almost
# entirely by removing the enforced friction cone. But the setup contains at
# least nine things that make the task easier than "push an object to a pose",
# and none of them had been costed. This sweep costs the ones that need
# training. THE BASELINES ARE REUSED, NOT RETRAINED: `full` (42248679 cells
# 0-2) is the control for Family A, `cross0` (cells 12-14) for Family B.
#
# PHASE 0 ALREADY RAN, at zero training cost, by replaying v28 checkpoints
# (logs/eval/v29_phase0/). Three scaffolds were costed without a single
# gradient step, and two of them came back NEGATIVE:
#
#   rotational damping 6.00 -> 3.12   0.739 -> 0.706   MATCHED, same 60 episodes
#   goal cone 30deg -> 90deg          0.739 -> 0.722   (different episodes)
#   portal 20cm -> 10cm, cross-room   0.615 -> 0.198   <- 3x COLLAPSE
#
# So `widecone` is NOT an arm here: a frozen policy already tolerates a 3x
# wider goal cone, and training with it off can only do better. The portal
# result is why Family B exists at all.
#
# ON THE DAMPING, which is the reason `physdamp` is still an arm despite the
# small replay effect. angular_drag_arm_cm is the lever arm in the table's
# rotational-friction torque, tau = mu*m*g*L. L is NOT free: for a body sliding
# on a plane it is the pressure-weighted mean radius of the contact patch. For
# this 10x6cm object, uniform pressure gives L = 3.12cm, and L = 5.83cm (the
# half-diagonal) is the ABSOLUTE CEILING -- all the load concentrated on the two
# farthest corners. The code has shipped 6.00, which no pressure distribution
# can produce. The comment called it "measured, not derived"; what was measured
# is that 1.0 spun the object out and 6.0 did not. The replay says it is not
# holding the RESULT up (-0.033 paired), but the value is still unphysical, so
# this arm asks the different question a replay cannot: does training under
# correct damping reach the same level?
#
# THE ARMS. Family A is same-room, single-factor from `full`. Family B is
# cross-room, single-factor from `cross0`.
#
#   nogapassist  gap_assist=false. _contact_frame_velocity forbids commanding
#                retreat faster than the object recedes -- it drags the finger
#                inward to chase a receding object (measured: a 0.1 push against
#                a 12cm/s recession is applied as 12cm/s). That is an ASSIST, not
#                physics; a real finger may back away. finger_velocity has never
#                had it, so this is the honest midpoint of full -> raw.
#   unmask       mask_inactive_finger=false. v27 found this harmful, but at 150k
#                WITH the cone on -- both since changed, so it is re-asked.
#   rawact       action_interface=finger_velocity. The (vx,vy) action space, no
#                face frame, no gap assist, no slip law. v25 measured raw 0.121
#                vs contact_frame 0.292 -- but with the cone on and at 150k.
#   randtheta    object_theta_spread_deg=90. The object has spawned axis-aligned
#                in 300/300 resets since the domain was built; +/-90deg covers
#                every distinct orientation of a rectangle. The face offset, the
#                face normal and the goal cone all rotate with it.
#   physdamp     angular_drag_arm_cm=3.12, the derived uniform-pressure value.
#   hardmode     all five at once. THIS IS THE NUMBER THAT MATTERS -- everything
#                else is attribution. If it lands near 0.7 the scaffolds were
#                scaffolding; near 0.2 and v28's result is mostly the setup.
#   narrowgap    portal y in [10,20]: a 10cm gap, exactly one object length, so
#                a broadside crossing has ZERO clearance. Frozen policies score
#                0.198 here; the question is whether training fixes it.
#   bigroom      90x60 board, wall at x=45, portal y in [10,50] -- the gap scaled
#                to keep the same 67% openness, so this isolates ROOM SIZE from
#                GAP SIZE. Today a room holds only 1.3 object-lengths of usable
#                width, which is why 12+cm same-room goals are 8% of episodes.
#                NOTE obs() scales positions by max(board_w, board_h), so this
#                arm also rescales every input -- which is exactly why it must be
#                trained rather than replayed.
#
# NOT IN THIS SWEEP, deliberately. The finger spawning already in contact on the
# correct face is a decision about where push ends and recontact begins, not a
# knob. Orientation GOALS need the goal space widened from (x,y) to (x,y,theta),
# which touches the HER buffer, and they are meaningless until randtheta lands.
#
# SCORING. Three passes, per docs/TODO.md:
#   1. common same-room benchmark, task keys pinned -> "did removing it cost?"
#   2. each arm on its OWN settings              -> "did it learn its own task?"
#   3. the BASELINE replayed under each arm's settings -> what the scaffold was
#      worth, at zero training cost. Pass 3 is the one that is easy to skip and
#      it is what caught v27's manufactured `place` win.
# gap_assist is an INTERFACE key (it changes what the outputs mean) and is read
# per cell; object_theta_spread_deg, angular_drag_arm_cm and the board geometry
# are TASK keys and are pinned. Expect the digest to move because keys were
# ADDED to the hash -- verify by diffing initial states, not by reading the hash.
# ===========================================================================

ARMS=(nogapassist unmask rawact randtheta physdamp hardmode narrowgap bigroom)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT_WIDE="portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]"
PORT_NARROW="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"
PORT_BIG="portals=[{x:45.0,y_lo:10.0,y_hi:50.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=true gap_assist=true"
TASK="push_cone_deg=30 require_settled=false disengaged_away_deg=60 push_range_min_cm=3.0 object_theta_spread_deg=null angular_drag_arm_cm=6.0 board_w_cm=50.0 board_h_cm=30.0"
PORT="${PORT_WIDE}"
SRG="same_room_goal_prob=1.0"

case "${ARM}" in
  nogapassist) IFACE="${IFACE/gap_assist=true/gap_assist=false}" ;;
  unmask)      IFACE="${IFACE/mask_inactive_finger=true/mask_inactive_finger=false}" ;;
  rawact)      IFACE="action_interface=finger_velocity mask_inactive_finger=true" ;;
  randtheta)   TASK="${TASK/object_theta_spread_deg=null/object_theta_spread_deg=90}" ;;
  physdamp)    TASK="${TASK/angular_drag_arm_cm=6.0/angular_drag_arm_cm=3.12}" ;;
  hardmode)
      IFACE="action_interface=finger_velocity mask_inactive_finger=false"
      TASK="${TASK/object_theta_spread_deg=null/object_theta_spread_deg=90}"
      TASK="${TASK/angular_drag_arm_cm=6.0/angular_drag_arm_cm=3.12}"
      ;;
  narrowgap)   PORT="${PORT_NARROW}"; SRG="same_room_goal_prob=0.0" ;;
  bigroom)
      PORT="${PORT_BIG}"; SRG="same_room_goal_prob=0.0"
      TASK="${TASK/board_w_cm=50.0 board_h_cm=30.0/board_w_cm=90.0 board_h_cm=60.0}"
      ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=400000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true ${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 target_clip=10 ${SRG} ${TASK} ${IFACE}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)";
  # GIT_COMMIT alone could not tell job 40944664 (pre-fix) from 40957220
  # (post-fix): both recorded a7c153a because the fix was uncommitted. Record
  # whether the tree was dirty and a hash of the diff, so a run's provenance
  # is checkable rather than a guess (docs/PROGRESS.md, v20).
  echo "GIT_DIRTY=$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)";
  echo "GIT_DIFF_SHA=$(git diff HEAD 2>/dev/null | sha256sum | cut -c1-16)"; } > "${RUN_DIR}/meta.txt"
git diff HEAD > "${RUN_DIR}/uncommitted.diff" 2>/dev/null
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
  wandb_group="scaffolds_${SLURM_ARRAY_JOB_ID}"
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
