#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=sweepB
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-06:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SWEEP B -- THE BUDGET AXIS, AND THREE VERDICTS RE-READ ON IT.
# 12 cells = 4 arms x seed{0,1,2} at 2.4M steps, ckpt_freq=600000.
#
# (Previous contents were job 44180162's Sweep A launcher, preserved per-run at
# logs/sweep_44180162/*/submit_script.sh. v33's is at logs/sweep_43679344/, v32's
# at logs/sweep_43572361/.)
#
# WHY THIS SWEEP, AND WHY IT IS NOT THE ONE THE OLD HEADER PROMISED.
# Sweep A reserved B for "the diversity arms, defined RELATIVE TO A3's winner".
# A3 did not win. Scored 2026-09-04 on both protocols, goals >=3cm, paired
# per-episode bootstrap CIs over the 48 in-scope episodes:
#
#   BUDGET, within-arm, same seeds, same protocol (249434216cd2), 1.2M - 600k
#     a1_v1_centre  +0.208 [+0.132,+0.285]      a2 +0.132 [+0.042,+0.222]
#     a3_v2_along   +0.125 [+0.035,+0.222]
#   SPAWN  A2 - A1     -0.035 / +0.021 (along)  -0.042 / -0.090* (centre)
#   OBS v2 A3 - A2     -0.021 / -0.132* (along) -0.062 / +0.021 (centre)
#                                    (* = 95% CI excludes zero; final / best)
#
# So the budget is the only effect in Sweep A that is significant on every arm
# and in the same direction, and it is 3-6x the size of either treatment. Both
# treatments came back null-or-negative with the sign flipping between
# checkpoints, which is the same "CI crosses zero" state v33's two steepest arms
# were in. THE ARM ORDERING ITSELF MOVES WITH BUDGET: on the centre protocol A2
# leads at 600k (0.653 vs A1's 0.618) and A1 leads at 1.2M (0.826 vs 0.785).
# A sweep read at one budget cannot rank arms here. That is the finding this
# sweep is built on, and it is why every arm below carries four rungs.
#
# CONTINUITY HELD, so the v33 table survives as a 600k table: A1 @600k scores
# 0.618 (0.604/0.667/0.583) on v33's own digest against v33 ctl's 0.674
# (0.604/0.750/0.667) -- indistinguishable at n=3. What does NOT survive is
# reading those numbers as converged prices.
#
# ---------------------------------------------------------------------------
# THE ARMS. Every one is A1's config with ONE key changed, so each cell's diff
# against ctl is a single line. A1 won Sweep A, so A1 is the control.
#
#   ctl        A1's config, verbatim, at 2.4M. Three jobs at once:
#              (a) DOES PUSH CONVERGE? A1's late diag slope over its last 300k
#                  measures +0.004 to +0.043 per 100k, positive on 9 of 9 cells
#                  -- halved from v33's +0.023..+0.055 at 600k but NOT zero. So
#                  1.2M is closer to converged than 600k and is not known to be
#                  converged, and pricing scaffolds there would repeat v33's
#                  mistake one rung later.
#              (b) THE REPLICATION CHECK. ctl @1.2M must reproduce A1 @1.2M's
#                  0.826 (per seed 0.792/0.833/0.854) and ctl @600k must
#                  reproduce A1 @600k's 0.618. Two independent anchor points, on
#                  the same protocol and the same code, at no extra cost. IF
#                  EITHER MISSES, STOP AND FIND OUT WHY BEFORE READING AN ARM.
#              (c) the control every price below is measured against, rung by
#                  rung.
#   obsv2      obs_version=2 normalize_goal_keys=true. THE ARM THAT GATES THE
#              REST OF THE PROJECT, which is why it is here instead of a
#              diversity arm. obs v2 exists to give every template ONE state+xi
#              head, and the composed system needs that head; but Sweep A
#              measured it costing push -0.132 [-0.222,-0.042] on model_best
#              while measuring it FREE for recontact (base_v2 0.967 against the
#              archived 0.978, and the two digests were confirmed to draw the
#              IDENTICAL 60 initial states -- see docs/PROGRESS.md). A verdict
#              that disagrees between two templates, at a budget that is known
#              to reorder arms, is not a verdict. Both keys are INTERFACE keys,
#              so this arm shares ctl's digest and needs no separate protocol.
#   widecone   push_cone_deg 30 -> 90. Re-reads v33's -0.069, whose diag slope
#              (+0.055/100k) was the steepest in that sweep and whose CI crossed
#              zero. TASK key: own digest, own floor, own-distribution scoring.
#   spread     object_theta_spread_deg null -> 90. Re-reads v33's -0.111, and it
#              is the ORIENTATION axis, which is the axis push is actually
#              graded on: 42 of 60 benchmark goals (70%) are already inside the
#              22.5deg tolerance at reset, success is 0.762 there against 0.500
#              on the 18 that must rotate, and mean |dtheta| INCREASES by
#              1.36deg per episode. Success is meanwhile flat at 0.833 across
#              the 3-6/6-9/9-12cm distance bins. TASK key: own digest, own
#              floor, own-distribution scoring.
#
# NOT IN THIS SWEEP, and each omission is a measurement rather than a budget cut:
#   freefinger  v33 priced mask_inactive_finger=false at -0.090, but the decision
#               to keep masking rests on the SAFETY column, not the success one:
#               forbidden_contact goes 2% -> 8-11% (v29), against a 19%
#               preregistered bar. A budget re-read cannot move that.
#   faceguard   MEASURED counterproductive to train under: ctl scores 0.604 on
#               the guarded task it never saw, against faceguard's own 0.542
#               (logs/eval/v33_faceguard_own/). The standing decision is to
#               enforce the face at EVAL. Re-scoring is minutes; an arm is 18h.
#   midaction   v33's -0.535, CI [-0.632,-0.438], the one verdict that was safe
#               at 600k and identical to v32's raw arm. Nothing to re-read.
#   the along-face spawn  Sweep A's A2. Null-or-negative on all four columns;
#               A1's config is the default and this sweep keeps it.
#   require_settled  Bar 2 was met ZERO-SHOT at 0.625. Re-score, never train.
#   the flat baseline  It is still not definable. Memo sec 5.2's decisive flat
#               arm is "identical reset distribution and action space, no
#               temporal hierarchy" -- and on a SINGLE-EDGE task that IS the push
#               option, so the comparison is empty until a composed task exists.
#               The composed task needs the offset door (the wall blocks the
#               straight path in 0 of 400 cross-room resets, so crossing is
#               currently a long straight push) and the Stage 1 ladder. Both are
#               builds, not cells. See docs/TODO.md.
#
# BUDGET AND WALL. 2.4M at Sweep A's measured 8.6h/1.2M is ~17.5h/cell; the wall
# is 30h. ckpt_freq=600000 puts snapshots at 600k/1.2M/1.8M on disk, so a cell
# killed at the wall still yields three rungs -- that is what saved v30 when it
# was cancelled at 3h55m. Aggregate for Eq 35 accounting: 12 x 2.4M = 28.8M
# environment interactions, plus 4 untrained floors at zero gradient steps.
#
# DIGESTS AND FLOORS. The common benchmark is A1's own CENTRE protocol,
# 249434216cd2 -- the digest every archived v32/v33/v34 push figure sits on, so
# ctl anchors to 0.674 (v33 @600k), 0.618 (A1 @600k) and 0.826 (A1 @1.2M) with no
# transfer step. finalize.sh must print 249434216cd2 for the PINS below; if it
# prints anything else, stop. Floors, all regenerated 2026-09-04 by
# tools/make_v35_floor.sh because a floor is specific to (interface, goal space,
# protocol) and never transfers:
#
#   logs/eval/v35_floor/ctl        obs v1, centre protocol      expect 0.042
#   logs/eval/v35_floor/obsv2      obs v2, centre protocol      its own floor
#   logs/eval/v35_floor/widecone   cone 90, own protocol        its own floor
#   logs/eval/v35_floor/spread     spread 90, own protocol      its own floor
#
# PRIMARY METRIC, pinned before the sweep and not to be changed after: mean
# success on goals >=3cm, under BOTH model and model_best, at each rung. The
# 5-bin mean has a 0.150 floor from the 0-3cm bin alone and is not the number.
# THE PUSH SUCCESS BARS ARE UNCHANGED and are not up for re-litigation: Bar 1 is
# >=0.40 on >=2 of 3 seeds with no bin at 0.00, and ctl already clears it at
# 0.826.
#
# PREREGISTERED VERDICTS, written before the numbers land:
#   |ctl @2.4M - ctl @1.2M| <= 0.05  -> push is converged; the prices at 2.4M are
#     the final prices and the fidelity-deviation list gets them.
#   ctl @2.4M - ctl @1.2M > +0.05    -> push is STILL not converged at 2.4M. Then
#     NO single number is a price, and the deliverable is the price-vs-budget
#     CURVE for each arm. Do not launch a third budget rung on the strength of
#     it -- report the curve and move to the ladder, which needs push graded,
#     not converged.
#   a scaffold price that SHRINKS toward zero across the four rungs -> v33 was
#     measuring the arm's slower convergence, not its cost. That is the
#     prediction on record for widecone and spread, both of which had the
#     steepest v33 slopes.
#   a price that HOLDS across all four rungs -> a real, converged cost, and it
#     goes in the deviation list with its number.
#   obsv2 penalty shrinking to within 0.05 of zero at 2.4M -> adopt obs v2
#     everywhere; the shared state+xi head is worth having and its apparent cost
#     was a budget artifact.
#   obsv2 penalty holding below -0.05 at 2.4M -> obs v2 costs push something
#     real, and the shared head needs a different design rather than a
#     different budget. Say so and price it; do not quietly keep both.
# ===========================================================================

set -e

ARMS=(ctl obsv2 widecone spread)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
# Byte-identical to Sweep A's A1 arm (logs/sweep_44180162/*a1_v1_centre_s0/
# meta.txt) apart from TOTAL_STEPS, so ctl is a replication and not a variant.
# The two ablated keys are pulled OUT of PROTO into their own variables so an
# arm overrides exactly one of them and the diff is readable.
PROTO="require_settled=false same_room_goal_prob=0.5 \
push_range_min_cm=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
push_range_max_cm=null push_spawn_along_frac=null"
GOAL="theta_tol_deg=22.5 theta_goal_window_deg=45.0"
CURRIC="curriculum_mode=band curriculum_levels=4 curriculum_threshold=0.6"

# Held FIXED across all four arms, and stated rather than defaulted.
FACE="guard_face=false"
MASK="mask_inactive_finger=true"
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 gap_assist=false"

# The three axes this sweep moves, one per arm. CONE and SPREAD are TASK keys
# (they move the reset/goal distribution and the digest); OBS is two INTERFACE
# keys (they change how the policy READS the world, not what the task is), which
# is why obsv2 scores on ctl's benchmark with no transfer step.
CONE="push_cone_deg=30"
SPREAD="object_theta_spread_deg=null"
OBS="obs_version=1 rich_obs=true normalize_goal_keys=false"

case "${ARM}" in
  ctl)      ;;
  obsv2)    OBS="obs_version=2 rich_obs=true normalize_goal_keys=true" ;;
  widecone) CONE="push_cone_deg=90" ;;
  spread)   SPREAD="object_theta_spread_deg=90" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=2400000
# 600000 exactly, so the rungs land on v33's budget (600k), Sweep A's (1.2M) and
# two beyond it. The budget axis is checkpoints, not arms.
CKPT_FREQ=600000
# 32, not the default 16: the advance gate compares a success rate against 0.6,
# and 16 episodes make that a 10-of-16 coin flip.
EVAL_EPS=32
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 \
target_clip=10 ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS} \
board_w_cm=50.0 board_h_cm=30.0 disengaged_away_deg=60 \
${PROTO} ${CONE} ${SPREAD} ${FACE} ${MASK} ${GOAL} ${CURRIC} ${IFACE} ${OBS}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

# THE BENCHMARK PROTOCOL, written next to the runs rather than retyped in the
# scorer -- a protocol that lives only in a launcher comment is what made every
# v25 cross-version comparison wrong, and a hardcoded portal in the scorer is
# what invalidated v33's first scoring run. curriculum_levels=null is the reverse
# sampler at full range, which is what every arm's last level trains on.
#
# THE COMMON PROTOCOL IS THE FACE-CENTRE ONE (push_spawn_along_frac=null), i.e.
# ctl's own task and v32/v33/v34-A1's, so every archived push figure is directly
# comparable with no transfer step. Expected digest: 249434216cd2.
# widecone and spread each train under a loosened TASK key, so each also needs
# the two-way treatment -- scored HERE (does loosened training still do the
# standard task?) and on its own distribution (is the loosened task learnable at
# all?). Given v33's faceguard result, the prediction on record is that ctl
# matches or beats the loosened arm on the loosened task in both cases.
# Written via mv, which is atomic: 12 tasks race here with identical content.
cat > "${SWEEP_DIR}/.PINS.$$" <<PINSEOF
use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 portal_arrival=false portal_goal=true portal_clearance_cm=0.5 guard_face=false rich_obs=true push_range_max_cm=null curriculum_mode=band curriculum_levels=null theta_tol_deg=22.5 theta_goal_window_deg=45.0 push_spawn_along_frac=null ${PORT}
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

rc=0
python train_contact.py \
  contact=${TEMPLATE} total_steps=${TOTAL_STEPS} seed=${SEED} \
  ${EXTRA_OVERRIDE} \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="sweepB_${SLURM_ARRAY_JOB_ID}" || rc=$?
# `|| rc=$?` and NOT a bare `rc=$?`: `set -e` is on, so a training
# failure would exit the task RIGHT HERE -- skipping the last-task-standing
# block below, which is what moves the staging logs and submits
# finalize.sh. A sweep whose last cell dies would then silently produce no
# scoring and leave its logs orphaned, which is how 621 staging files
# accumulated. rc is initialised so the trap still reports success.

# Am I the last task standing? `-o "%i"`, NOT "%A_%a": on this Slurm %a renders
# as the ACCOUNT name ("43892866_hankyang_lab"), so the grep matched nothing,
# `still` never reached 0, and this whole block silently never ran on v29, v32
# or v33. It fired for the first time on v34 -- and killed its own scorer,
# because it writes slurm_logs/ INTO the sweep dir and score_sweep.cell_dirs
# globbed */ (fixed 2026-09-04, gated in test_code.py static).
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
  # its own 30h wall, and scoring 60 checkpoints is not free. finalize.sh scores
  # model.zip and model_best.zip; the three intermediate rungs are scored by
  # tools/score_v35_rungs.sh, which is the budget axis and the point of the sweep.
  sbatch slurm/finalize.sh "${SWEEP_DIR}" sweepB
  sbatch tools/score_v35_rungs.sh "${SWEEP_DIR}"
fi
exit $rc
