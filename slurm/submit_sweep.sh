#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_abl
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-17
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v33 SCAFFOLD-REMOVAL SWEEP -- 18 cells = 6 arms x seed{0,1,2} at 600k steps.
#
# (Previous contents were job 43572361's v32 launcher, preserved per-run at
# logs/sweep_43572361/*/submit_script.sh -- all 11 copies verified identical to
# the file overwritten here.)
#
# THE QUESTION. v32 established that push is learnable with a reverse
# curriculum: curric 0.674 mean on goals >=3cm against a 0.042 untrained floor,
# beating the no-curriculum control 0.583 on 3/3 seeds. But it is learnable on a
# task made easier than the memo's push option in several identifiable ways.
# This sweep removes them ONE AT A TIME and prices each one.
#
# Every arm is v32's `curric` with exactly ONE key changed. The curriculum stays
# ON in all six -- it is now the working configuration, not a variable.
#
#   ctl         nothing changed. v32 `curric` rerun on THIS code tree.
#   freefinger  mask_inactive_finger=false. The memo has two independent
#               fingertips; v32 froze the idle one. v29 measured unmasking a
#               wash on success and 4x worse on forbidden_contact, but that was
#               without a curriculum.
#   widecone    push_cone_deg 30 -> 90. The goal stops being nearly straight
#               ahead of the contacted face. v30 measured 30->90 costing
#               0.822->0.767 and 30->180 costing ->0.561, without a curriculum.
#   spread      object_theta_spread_deg null -> 90. The object stops spawning
#               axis-aligned. v29 measured -0.044.
#   faceguard   guard_face=adjacent. See below -- this is the big one.
#   midaction   finger_velocity + restrict_contact_actions=true. The middle rung
#               between the contact frame (v32 base 0.583) and raw velocity
#               (v32 raw 0.139), separating "the frame helps" from "forbidding
#               retreat helps". v18 measured the clamp alone as no help: 62% of
#               contact breaks are TANGENTIAL slides off a corner the clamp
#               never restricts. New here only because the curriculum is on.
#
# WHY ctl IS RERUN RATHER THAN REUSED. This tree changed contact_templates.py
# and gym_env.py for the face guard. The change is flag-gated and the v32 env
# digest 249434216cd2 is reproduced exactly (that is what folding the new mode
# into `guard_face` instead of adding a second key bought), so reusing v32's
# numbers would be defensible -- but three cells is cheaper than the argument,
# and it re-checks that the v32 curric result reproduces at all.
#
# TWO KEY CLASSES, TWO SCORING TREATMENTS. This is the part that decides whether
# the results mean anything.
#   freefinger and midaction change INTERFACE keys, which sit OUTSIDE the env
#   digest and are read per cell. Same benchmark as ctl, directly comparable.
#   widecone, spread and faceguard change TASK keys, so each gets its own
#   digest. slurm/finalize.sh scores every arm under the COMMON tight protocol
#   in PINS.txt -- does the loosened training still do the standard task?
#   Scoring them a second time on their OWN distribution -- is the loosened task
#   learnable at all -- is a separate deliberate step, not automatic, because
#   which override belongs to which arm is a judgement call.
#
# THE FACE GUARD, AND A MEASUREMENT THAT CHANGED THE DESIGN.
# Eq 7 makes the contact face an edge parameter and Eq 40's chi_push enforces
# it, so a policy that walks onto another face has executed a different edge
# than the one it was labelled with. v32's two best policies were scored under
# guard_face=true with nothing else changed (logs/eval/v32_faceprobe):
#
#   push_curric_s1   guard_face=false   0.683   arrived 41, horizon 14, lost 5
#                    guard_face=true    0.083   WRONG_FACE 49/60 (81.7%)
#   push_base_s1     guard_face=false   0.600
#                    guard_face=true    0.083   WRONG_FACE 52/60 (86.7%)
#
# So face switching is not a rare corner-rounding artifact: it is how these
# policies push, on ~82-87% of episodes, at a median of 12 ticks. An earlier
# reading that the 4.0cm contact-loss budget bounds it was WRONG -- that budget
# applies only to time spent NOT touching, and a finger that keeps contact can
# slide along the surface indefinitely. The real number is geometric: the finger
# starts at the face centre of a 10x6 object and the corner is
# |ly| = 3*5.5/5 = 3.3cm away, which is ~4 ticks at 20cm/s.
#
# Splitting those violations by WHICH face, same policies, guard_face=adjacent:
#
#   push_curric_s1   0.683 unguarded -> 0.483   wrong_face 13/60 (21.7%)
#   push_base_s1     0.600 unguarded -> 0.583   wrong_face  9/60 (15.0%)
#
# So most of the 82-87% is one corner rounded onto an ADJACENT face, and only
# 15-22% of episodes reach the OPPOSITE face -- the transition that actually
# breaks the edge label, since pushing from the far side is the reverse of the
# edge that was named. That is why this arm runs `adjacent` and not `strict`:
# strict is unlearnable-shaped (v31 took 9/9 cells to 0.000), while adjacent
# starts from 0.483-0.583 ZERO-SHOT on policies that never saw the constraint.
# Those two numbers are the arm's FLOOR, not a prediction -- a policy trained
# under the guard can learn to avoid the corner it currently rounds.
#
# ONE FINDING THIS ALREADY FORCES INTO THE RECORD. Unguarded, curric beats base
# 0.683 to 0.600. Under the face guard the ordering INVERTS, 0.483 to 0.583. So
# part of the curriculum's v32 advantage is bought with behaviour that violates
# the edge label it was executing. v32's headline stands as "the curriculum
# helps on the task as scored" and does NOT stand as "the curriculum learns a
# better push option". This arm is the test of which one is true.
#
# UNCHANGED FROM v32, and why: pure sparse reward (all w_*=0, goal_reward=10 on
# arrival, arrival terminates, so Q* <= 10 and target_clip=10 is a provable
# bound); same_room_goal_prob=0.5 with portal_goal (the doorway POSE is the
# proxy for entering the next region, which keeps ONE definition of success so
# HER can relabel toward it); theta_tol_deg=22.5 (Eq 11's eight orientation
# bins); push_range_min_cm=null; angular_drag_arm_cm=3.12; band curriculum with
# 4 levels and threshold 0.6, windows (0.00,0.35) (0.15,0.60) (0.35,0.85)
# (0.00,1.00) as fractions of each edge's reachable range.
#
# THE CONTROL SHARES THE SAMPLER, still: every arm runs curriculum_mode=band.
#
# UNTRAINED FLOOR on the common protocol, goals >=3cm: 0.042 restricted actions,
# 0.000 raw (logs/eval/v32_floor). The digest there must match the digest
# finalize.sh prints, or nothing below is anchored.
#
# PREREGISTERED VERDICTS -- written before the sweep, not after. All on goals
# >=3cm, under BOTH model and model_best, on >=2 of 3 seeds.
#   A scaffold is CHEAP if its arm lands within 0.05 of ctl. Remove it for good.
#   A scaffold is LOAD-BEARING if its arm drops more than 0.15 below ctl. It
#     stays, and it goes in the recorded fidelity-deviation list WITH its price.
#   In between: report the number, no verdict, do not re-run hoping it moves.
#   faceguard is the exception. Its own 0.042 floor is the bar, not ctl:
#     untrained-under-the-guard is 0.083 with no training at all, so anything
#     that does not clearly beat 0.083 means the constraint is unlearnable at
#     this horizon rather than merely expensive.
#
# NOT IN THIS SWEEP, deliberately:
#   require_settled -- excluded by explicit instruction this round. Still real
#     spec (Eq 13) and still what makes push composable; its own arm later.
#   The flat (non-hierarchical) baseline. It is the actual blocker for any
#     hierarchy claim and it must inherit EVERY scaffold left standing after
#     this sweep, which is why it comes after, not before.
#   The adaptive window (Florensa's real rule: hold the window where success
#     sits between ~10% and ~90%). More machinery; the fixed schedule works.
# ===========================================================================

set -e

ARMS=(ctl freefinger widecone spread faceguard midaction)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
# The five ablated keys are pulled OUT of PROTO into their own variables so an
# arm overrides exactly one of them and the diff is readable.
PROTO="require_settled=false same_room_goal_prob=0.5 \
push_range_min_cm=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
rich_obs=true push_range_max_cm=null"
GOAL="theta_tol_deg=22.5 theta_goal_window_deg=45.0"
CURRIC="curriculum_mode=band curriculum_levels=4 curriculum_threshold=0.6"

CONE="push_cone_deg=30"
SPREAD="object_theta_spread_deg=null"
FACE="guard_face=false"
MASK="mask_inactive_finger=true"
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 gap_assist=false"

case "${ARM}" in
  ctl)        ;;
  freefinger) MASK="mask_inactive_finger=false" ;;
  widecone)   CONE="push_cone_deg=90" ;;
  spread)     SPREAD="object_theta_spread_deg=90" ;;
  faceguard)  FACE="guard_face=adjacent" ;;
  # gap_assist is inert under finger_velocity (it only feeds ContactFrameCommand)
  # but is stated anyway: it is an INTERFACE key and lands in every eval's
  # recorded interface dict, so leaving it to the config default makes two arms
  # look like they differ on it when they do not.
  midaction)  IFACE="action_interface=finger_velocity restrict_contact_actions=true gap_assist=false" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=600000
CKPT_FREQ=200000
# 32, not the default 16: the advance gate compares a success rate against 0.6,
# and 16 episodes make that a 10-of-16 coin flip.
EVAL_EPS=32
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 \
target_clip=10 ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS} \
board_w_cm=50.0 board_h_cm=30.0 disengaged_away_deg=60 \
${PROTO} ${CONE} ${SPREAD} ${FACE} ${MASK} ${GOAL} ${CURRIC} ${IFACE}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

# THE BENCHMARK PROTOCOL, written next to the runs rather than retyped in the
# scorer -- a protocol that lives only in a launcher comment is what made every
# v25 cross-version comparison wrong. These are the CONTROL's task keys with
# curriculum_levels=null (the reverse sampler at full range, which is what every
# arm's last level trains on), so every arm is scored on one common task.
# Written via mv, which is atomic: 18 tasks race here with identical content.
cat > "${SWEEP_DIR}/.PINS.$$" <<PINSEOF
use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 portal_arrival=false portal_goal=true portal_clearance_cm=0.5 guard_face=false rich_obs=true push_range_max_cm=null curriculum_mode=band curriculum_levels=null theta_tol_deg=22.5 theta_goal_window_deg=45.0 ${PORT}
PINSEOF
mv -f "${SWEEP_DIR}/.PINS.$$" "${SWEEP_DIR}/PINS.txt"

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
  wandb_group="abl_${SLURM_ARRAY_JOB_ID}"
rc=$?

# Am I the last task standing? `-o "%i"`, NOT "%A_%a": on this Slurm %a renders
# as the ACCOUNT name ("43892866_hankyang_lab"), so the grep matched nothing,
# `still` never reached 0, and this whole block silently never ran on v29, v32
# or v33 -- verified by three sweeps with no slurm_logs/ and 36 orphaned staging
# files. Measured with a 3-task probe job, not reasoned about.
still=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}" -t PENDING,RUNNING -o "%i" \
        | grep -v "^${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}$" | wc -l)
# mkdir is atomic, so two tasks finishing in the same instant cannot both
# submit finalize.
if [ "${still}" -eq 0 ] && mkdir "${SWEEP_DIR}/.finalized" 2>/dev/null; then
  mkdir -p "${SWEEP_DIR}/slurm_logs"
  find logs/slurm_staging -maxdepth 1 -type f \
       \( -name "${SLURM_ARRAY_JOB_ID}_*.out" -o -name "${SLURM_ARRAY_JOB_ID}_*.err" \) \
       -exec mv -t "${SWEEP_DIR}/slurm_logs/" {} + 2>/dev/null
  # Score + render, as a follow-on job rather than inline: this cell may be near
  # its own 8h wall, and scoring 36 checkpoints is not free.
  sbatch slurm/finalize.sh "${SWEEP_DIR}"
fi
exit $rc
