#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=sweepA
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-14:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SWEEP A -- THE NEW BASELINE. 9 cells = 3 arms x seed{0,1,2} at 1.2M steps.
#
# (Previous contents were job 43679344's v33 scaffold-removal launcher,
# preserved per-run at logs/sweep_43679344/*/submit_script.sh. v32's is at
# logs/sweep_43572361/*/submit_script.sh.)
#
# THE QUESTION, and it is three questions read off nine cells. Every arm runs
# 1.2M steps with ckpt_freq=600000, so the 600k snapshot gives the budget axis
# for free rather than costing three more arms:
#
#   A1  obs v1, raw goal keys, face-CENTRE spawn      the continuity anchor
#   A2  obs v1, raw goal keys, along-face spawn       prices the SPAWN alone
#   A3  obs v2 + normalized goal keys, along-face     the protocol B/C/D inherit
#
#   budget    600k checkpoint vs 1.2M final, WITHIN each arm
#   spawn     A2 - A1
#   obs v2    A3 - A2
#   continuity A1 @600k should land near v33 ctl's 0.674
#
# WHY THE BUDGET AXIS IS FIRST. Every one of v33's 18 cells was STILL IMPROVING
# at 600k -- measured diag-eval slope per 100k over the last half of training:
# widecone +0.0553, freefinger +0.0437, faceguard +0.0306, ctl +0.0266, spread
# +0.0250, midaction +0.0225, and the argmax sat at 555k-595k of 600k for every
# arm. The two STEEPEST are exactly the two whose paired CIs cross zero. So v33
# priced its scaffolds at a budget where nothing had converged, which is v28's
# mistake at a larger scale: there, 150k -> 400k bought the working arm +0.45 and
# the friction-cone arm only +0.09, and the short sweep read the cone's cost as
# 0.126 against its true 0.46.
#
#   IF A1 @1.2M LANDS MATERIALLY ABOVE 0.674, EVERY v33 SCAFFOLD PRICE NEEDS
#   RE-READING. `midaction` (-0.535, slower slope, CI [-0.632, -0.438]) is the
#   only v33 verdict that was safe at 600k.
#
# WHY THE SPAWN IS ITS OWN ARM RATHER THAN FOLDED INTO A3. A contact at the face
# CENTRE pushing along the inward normal produces EXACTLY ZERO torque -- the
# lever arm is parallel to the force. That is why push's net rotation measures a
# median 1.8deg/episode against a +/-45deg goal window, and it is why the spawn
# GATES the orientation-diversity arm in Sweep B. Folding it into the obs change
# would make two unrelated effects inseparable.
#
# ZERO-SHOT, already measured, NOT a prediction: v33 ctl_s1 scores 0.583 on the
# randomized-spawn protocol against 0.683 on its own. A2 turns that into a
# trained number.
#
# UNTRAINED FLOOR: logs/eval/v34_floor, and it is 0.042 on goals >=3cm for ALL
# THREE contact_frame configs -- the spawn does not move it. So THE PUSH SUCCESS
# BAR IS UNCHANGED (docs/TODO.md, Bar 1 >=0.40, ~10x the floor) and is not up for
# re-litigation. Raw actions floor at 0.000, which is what Sweep C's ppo_raw arm
# is measured against. The a1_v1_centre floor cell reproduces
# logs/eval/v32_floor/untrained_pose_contact_frame exactly, so the archived
# anchor survived every change between v33 and here.
#
# DIGESTS. obs_version and normalize_goal_keys are INTERFACE keys, so they do NOT
# move the digest: A2 and A3 share 646ba4ae1fd4 and are directly comparable.
# A1 sits on 249434216cd2, v33's own digest, because a post-hoc TASK key at its
# default is omitted from the stamp -- which is what keeps A1 comparable to
# every archived v32/v33 number. finalize.sh must print 646ba4ae1fd4 for the
# PINS below; if it prints anything else, stop.
#
# PRIMARY METRIC, pinned before the sweep: mean success on goals >=3cm, under
# BOTH model and model_best. The 5-bin mean has a 0.150 floor from the 0-3cm bin
# alone and is not the number.
#
# NOT IN THIS SWEEP, deliberately:
#   The diversity arms (Sweep B). They are defined RELATIVE to A3's winner, so
#     they cannot be launched until this reads out.
#   PPO (Sweep C) and recontact (Sweep D). Same reason for C; D is a different
#     template with its own launcher.
#   require_settled -- NO LONGER NEEDED AS AN ARM. Bar 2 was MET zero-shot on
#     2026-09-03: v33's frozen ctl checkpoints score 0.625 on goals >=3cm under
#     require_settled=true against 0.674 position-only and a 0.000 settled
#     floor, so settling costs 0.049 and needs no training. See
#     logs/eval/v34_bar2/ and tools/bar2_zeroshot.sh. Every arm here keeps
#     require_settled=false so it stays comparable to v33; re-score with
#     tools/bar2_zeroshot.sh afterwards, which is minutes, not an arm.
#   The flat baseline. It must inherit whatever scaffolds survive, so it comes
#     after, not before.
# ===========================================================================

set -e

ARMS=(a1_v1_centre a2_v1_along a3_v2_along)
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
push_range_max_cm=null"
GOAL="theta_tol_deg=22.5 theta_goal_window_deg=45.0"
CURRIC="curriculum_mode=band curriculum_levels=4 curriculum_threshold=0.6"

# Held FIXED across all three arms, and stated rather than defaulted: these are
# the scaffolds v33 priced and none of them is what this sweep is asking about.
CONE="push_cone_deg=30"
SPREAD="object_theta_spread_deg=null"
FACE="guard_face=false"
MASK="mask_inactive_finger=true"
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 gap_assist=false"

# The two axes this sweep moves. SPAWN is a TASK key (it moves the reset
# distribution and the digest); OBS is two INTERFACE keys (they change how the
# policy READS the world, not what the task is), which is exactly why A2 and A3
# score on one benchmark.
SPAWN="push_spawn_along_frac=null"
OBS="obs_version=1 rich_obs=true normalize_goal_keys=false"

case "${ARM}" in
  a1_v1_centre) ;;
  a2_v1_along)  SPAWN="push_spawn_along_frac=0.7" ;;
  a3_v2_along)  SPAWN="push_spawn_along_frac=0.7"
                OBS="obs_version=2 rich_obs=true normalize_goal_keys=true" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=1200000
# 600000 exactly, so the mid-run snapshot IS v33's budget and the budget axis is
# a checkpoint rather than three more arms.
CKPT_FREQ=600000
# 32, not the default 16: the advance gate compares a success rate against 0.6,
# and 16 episodes make that a 10-of-16 coin flip.
EVAL_EPS=32
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 \
target_clip=10 ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS} \
board_w_cm=50.0 board_h_cm=30.0 disengaged_away_deg=60 \
${PROTO} ${CONE} ${SPREAD} ${FACE} ${MASK} ${GOAL} ${CURRIC} ${IFACE} \
${SPAWN} ${OBS}"
RUN_TAG="push_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

# THE BENCHMARK PROTOCOL, written next to the runs rather than retyped in the
# scorer -- a protocol that lives only in a launcher comment is what made every
# v25 cross-version comparison wrong, and a hardcoded portal in the scorer is
# what invalidated v33's first scoring run. curriculum_levels=null is the reverse
# sampler at full range, which is what every arm's last level trains on.
#
# THE COMMON PROTOCOL IS THE ALONG-FACE ONE (push_spawn_along_frac=0.7), i.e.
# A2/A3's task, not A1's. Two of three arms train there and it is the protocol
# Sweeps B/C/D inherit, so it is the one worth being exact about. A1 therefore
# gets the v33 two-way treatment: scored HERE (does a centre-trained policy do
# the randomized task?) and, as a deliberate second step, on its own
# 249434216cd2 protocol, which is what makes it comparable to every archived
# v32/v33 number. Expected digest here: 646ba4ae1fd4.
# Written via mv, which is atomic: 9 tasks race here with identical content.
cat > "${SWEEP_DIR}/.PINS.$$" <<PINSEOF
use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 portal_arrival=false portal_goal=true portal_clearance_cm=0.5 guard_face=false rich_obs=true push_range_max_cm=null curriculum_mode=band curriculum_levels=null theta_tol_deg=22.5 theta_goal_window_deg=45.0 push_spawn_along_frac=0.7 ${PORT}
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
  wandb_group="sweepA_${SLURM_ARRAY_JOB_ID}" || rc=$?
# `|| rc=$?` and NOT a bare `rc=$?`: `set -e` is on, so a training
# failure would exit the task RIGHT HERE -- skipping the last-task-standing
# block below, which is what moves the staging logs and submits
# finalize.sh. A sweep whose last cell dies would then silently produce no
# scoring and leave its logs orphaned, which is how 621 staging files
# accumulated. rc is initialised so the trap still reports success.

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
  sbatch slurm/finalize.sh "${SWEEP_DIR}" sweepA
fi
exit $rc
