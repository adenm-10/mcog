#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_contact
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-04:00:00
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v26 PUSH sweep: the free finger, and a friction-consistent slip law.
#
# 15 cells = arm {base, legacy, unmask, place, unmask_place} x seed {0,1,2}.
# 150k steps each, contact_frame throughout.
#
# (The previous contents were job 42007967's launcher. All 16 of its per-run
# copies, logs/sweep_42007967/*/submit_script.sh, are byte-identical to the
# copy overwritten here -- verified before overwriting.)
#
# WHAT v25 ESTABLISHED. contact_frame roughly doubled push success and produced
# the project's first nonzero 12+cm result. Best cell 0.417. Scoring all 16
# cells needed each cell's OWN interface read from its meta.txt; scoring them
# all as finger_velocity inverted the result once already (docs/PROGRESS.md v25).
#
# TWO CHANGES ARE UNDER TEST HERE.
#
# 1. THE FREE FINGER. gym_env.step zeroed the non-pushing finger's action, which
#    does not remove it -- it leaves it SERVO-HELD wherever it spawned. Harmless
#    when a push moved 2cm; not once pushes moved 6-10cm. Measured:
#    forbidden_contact went 8% -> 17-19% and started firing LATE (tick 26-31 vs
#    8-9), which is the object arriving at a stationary fingertip, not a bad
#    grasp at reset. Two independent fixes, hence two axes:
#      unmask = mask_inactive_finger=false. The policy controls both fingers and
#               the Eq 40 guard is the only thing keeping the free one clear.
#               This is the option the masking comment warned against (its
#               exploration noise alone used to trip the guard), so it is a real
#               test of whether that warning still holds at 150k steps.
#      place  = disengaged_away_deg=60. Spawn the free finger in a 60deg cone
#               centred on the ACTIVE face's outward normal, i.e. behind the
#               object's travel. Sidesteps the collision instead of solving it.
#    ZERO-TRAINING FALSIFIER, ALREADY PASSED. Replaying the v25 SOTA checkpoint
#    (cell 15, frozen policy) under the new spawn cone: forbidden_contact
#    8/60 (13.3%) -> 1/60 (1.7%). The mechanism is confirmed. Overall success
#    fell 0.433 -> 0.333 in that same replay, but the policy was trained on the
#    uniform ring and is off-distribution there, and 26-vs-20 arrivals in 60 is
#    ~1.6 standard errors. Whether placement HELPS is what these cells decide.
#
# 2. SLIP IS NOW DERIVED, NOT TUNED. slip_limit was a free tangential ceiling
#    picked by sweeping. It is now Coulomb: friction_cone sets the tangential
#    budget to mu * push * v_max with mu = finger_friction, the same coefficient
#    pymunk already uses for that contact. It scales with the normal push and
#    vanishes when the finger stops pressing, which a fixed ceiling does not.
#    The worst-case command sits exactly on arctan(mu) = 36.87deg off the face
#    normal (asserted in test_code.py contact).
#      legacy = speed_fraction slip_limit=1.0, v25's best setting, kept as ONE
#               comparison arm so we know what adopting the physical constraint
#               cost. This is not a slip sweep and slip is not being tuned again.
#
# WHAT E3 KILLED, BEFORE IT WAS RUN. The planned horizon/require_settled sweep
# assumed push failures OVERSHOOT. eval_contact.py now reports closest approach
# per episode, and on the SOTA checkpoint 0% of the 34 non-arrivals ever came
# within arrival_eps, only 20-24% within 1cm, median closest approach 3.75cm,
# and mean (final - min) just 1.40cm. Failures never get close. That is an
# AIMING problem, so settling cannot fix it and a longer horizon cannot either
# (median closest approach is tick 50 of 200 -- they get nearest halfway and
# then wander). Those 12 cells are not being run.
#
# PREREGISTERED PREDICTIONS:
#  - place cuts forbidden_contact to ~2%. This is near-certain; the frozen-policy
#    replay already showed it. The open question is whether success follows, and
#    if forbidden_contact falls while success does NOT, the guard was catching
#    episodes that were failing anyway and E1 is cosmetic.
#  - unmask is the risky arm. If forbidden_contact rises above the 17-19%
#    baseline, the free finger's exploration noise is still net-harmful and
#    masking should stay. If it falls, the policy learned to park it clear and
#    unmask strictly dominates place, which only moves the spawn.
#  - unmask_place tests whether they compose or are redundant. Redundancy is the
#    likelier outcome: a policy that can move the finger does not need it spawned
#    out of the way.
#  - legacy vs base isolates the slip law. friction_cone is STRICTER (budget
#    0.75*push*v_max <= 15 cm/s vs a flat 20 cm/s), so a loss means tangential
#    authority beyond the friction cone was doing real work -- worth knowing,
#    since a real finger would not have it. No difference is the expected result.
#  - FALSIFIER FOR THE WHOLE ROUND: if no arm beats base outside seed spread,
#    the free finger was not push's binding constraint and the next move is the
#    aiming problem E3 exposed, not a fifth fix to the contact interface.
#
# NOT changed, deliberately: target_clip stays null (push's critic is already
# calibrated, measured gap +1.77 against the provable bound of 10). Reward stays
# fully sparse, every w_* zero -- the guard stays terminal-only, with no w_m
# term, so `unmask` measures the guard alone. push_cone_deg stays 30 and
# same_room_goal_prob stays 1.0 in every cell, so the sampler is held fixed.
#
# SCORING -- READ THIS, IT HAS BITTEN THIS PROJECT TWICE.
# Two kinds of key, and they are handled OPPOSITELY:
#   INTERFACE keys (action_interface, slip_model, slip_limit,
#   restrict_contact_actions, mask_inactive_finger) must be taken FROM EACH
#   CELL -- they decide what the policy's numbers mean. eval_contact.py excludes
#   them from the digest for exactly this reason.
#   TASK keys (require_settled, horizon, sampler, disengaged_away_deg) must be
#   pinned at the protocol value for every cell, or the arms are not measuring
#   the same success. The digest contains them, so drift is caught.
#
# THE ARMS SPLIT INTO TWO DIGEST GROUPS, because disengaged_away_deg is a task
# key and placement genuinely changes the reset distribution:
#   group A (uniform ring):  base, legacy, unmask
#   group B (60deg cone):    place, unmask_place
# Within a group the comparison is digest-exact. ACROSS groups it is not: both
# are stratified to identical distance bins so d0 is matched by construction
# (which was v22's failure), but the reset distributions differ on purpose.
# To make the cross-group read interpretable, also score group A's `base`
# checkpoints under group B's overrides -- no extra training, one eval each --
# giving the transfer number that separates "placement helped" from "placement
# made an easier task".
#
#   python eval_contact.py contact=push seed=0 \
#     use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
#     board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] \
#     min_progress_ticks=1 learning_starts=10000 require_settled=false \
#     push_cone_deg=30 same_room_goal_prob=1.0 \
#     <the cell's own interface keys, from its meta.txt> \
#     <the group's disengaged_away_deg> \
#     eval_ckpt=<cell>/model_best.zip
# The printed env digest must match across every cell in a group.
# ===========================================================================

ARMS=(base legacy unmask place unmask_place); SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

CF="action_interface=contact_frame"
case "${ARM}" in
  base)         ARM_OVERRIDE="${CF} slip_model=friction_cone"                                                        ;;
  legacy)       ARM_OVERRIDE="${CF} slip_model=speed_fraction slip_limit=1.0"                                        ;;
  unmask)       ARM_OVERRIDE="${CF} slip_model=friction_cone mask_inactive_finger=false"                             ;;
  place)        ARM_OVERRIDE="${CF} slip_model=friction_cone disengaged_away_deg=60"                                 ;;
  unmask_place) ARM_OVERRIDE="${CF} slip_model=friction_cone mask_inactive_finger=false disengaged_away_deg=60"      ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=150000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 same_room_goal_prob=1.0 push_cone_deg=30 require_settled=false target_clip=null ${ARM_OVERRIDE}"
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
  wandb_group="free_finger_${SLURM_ARRAY_JOB_ID}"
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
