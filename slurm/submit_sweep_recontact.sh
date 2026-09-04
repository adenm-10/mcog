#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=recon_sweep
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-10:00:00
#SBATCH --array=0-11%4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SWEEP D -- RECONTACT AT OBS v2, AND THE FIRST INTERPRETABLE GAMMA NUMBERS.
#
#   sbatch slurm/submit_sweep_recontact.sh                      # GAMMA, 12 cells
#   sbatch --array=0-2 --export=ALL,GROUP=base \
#          slurm/submit_sweep_recontact.sh                       # BASE, 3 cells
#
# TWO GROUPS, TWO SBATCH CALLS, TWO SWEEP DIRS, TWO PINS FILES -- deliberately.
# The 2-D single-finger goal and Eq 13's 6-D interface goal are different GOAL
# SPACES, so SB3's check_for_correct_spaces refuses to load one against the
# other's env. They can never share a benchmark, so pretending they are one
# sweep would only produce a PINS.txt that is wrong for three of the cells.
#
# WHY THIS SWEEP EXISTS. Recontact has exactly one working policy and it solves
# the EASIEST version of the task: `recon_base` is gamma_goal=false, i.e. ONE
# fingertip to a point just outside one randomly chosen face, with the other
# finger ignored. Measured 2026-09-02 on the stratified 60-episode protocol at
# digest a78252c0a0a6: 0.967 / 0.983 / 0.983, mean 0.978, no floored bin.
#
# What does NOT work is the thing composition actually needs -- Eq 13's
# canonical interface Gamma_l, both fingertips plus their contact states. v31's
# recon_goal/recon_full scored ~0.000 on 6 of 9 cells. Those numbers are also
# UNINTERPRETABLE: `step` scored arrival with target=_goal_xy[:2], which under a
# Gamma goal is finger L's slot, while measuring whichever finger was ACTIVE.
# Active is R about half the time, and a state perfectly achieving the intended
# 6-D goal was scored arrived on only 254/500 (50.8%) of resets. Fixed
# 2026-09-02 (docs/PROGRESS.md P1-P4); ~63 GPU-hours need re-running, not
# re-scoring. THIS SWEEP IS THAT RE-RUN.
#
# ---------------------------------------------------------------------------
# GROUP=base -- 3 cells, 3 seeds. Does obs v2 COST recontact anything?
#
# This is a replication, so it changes obs and NOTHING else. In particular it
# pins angular_drag_arm_cm=6.0, the value recon_base actually trained on, even
# though 6.0 is above the physically attainable ceiling (5.83cm, the object's
# half-diagonal) and 3.12 is now the config default. A replication that
# silently inherited the better constant would not be a replication.
#
# It also keeps recontact.yaml's shaping (w_T=0.02 w_a=0.01 w_m=2.0). Recontact
# has NEVER been pure sparse; zeroing it here would confound the obs question.
#
# NOT a free replication: obs v2 takes recontact from 17 to 38 dims and hands
# it contact normals, force, wall distances and xi -- features it has never
# had. Wall distances are near-useless with the object pinned at board centre.
# If base_v2 lands below 0.978 that is a real obs-v2 cost, not noise.
#
# ---------------------------------------------------------------------------
# GROUP=gamma -- 12 cells, 4 arms x 3 seeds. Eq 13, with the bug fixed.
#
#   gamma_free          fingers start DISENGAGED. The canonical acquisition
#                       task, and what a composed push->recontact handoff
#                       actually has to do. Pure sparse.
#   gamma_init          fingers start in a RANDOM COMPLIANT interface drawn from
#                       the same class table. Pure sparse. This is the
#                       CURRICULUM arm, not a convenience: v31 traced the Gamma
#                       failure to a 4-way conjunction that essentially never
#                       fires -- L inside its 0.3cm anchor tolerance in 1/60
#                       episodes, R inside 2.0cm in 2/60, both touch flags
#                       matching in 4/60. Starting IN an interface is a much
#                       shorter trip, so it is the only arm with a plausible
#                       path to nonzero.
#   gamma_init_noclip   gamma_init with target_clip=null and NOTHING else
#                       changed. Pure sparse. This arm exists because shaping
#                       FORCES target_clip off, so the shaped arm below moves
#                       two things at once -- and the one it moves for free is
#                       the one that took recontact from 1/6 to 6/6 seeds.
#                       gamma_init - gamma_init_noclip prices the clamp.
#   gamma_init_shaped   gamma_init_noclip plus the dense terms and per-outcome
#                       guard penalties. target_clip=null, enforced by
#                       train_contact: the [0, clip] bound assumes
#                       Q* <= goal_reward, which a per-tick penalty breaks.
#                       Read against gamma_init_noclip, which is what makes the
#                       shaping the only difference.
#
# init_gamma_modes IS A TASK KEY: it changes the reset distribution. So
# gamma_init/gamma_init_shaped are separate EXPERIMENTS from gamma_free, and
# PINS below pins the COMMON protocol at init_gamma_modes=[free] -- the harder,
# canonical task. Each starting-in-an-interface arm therefore needs the v33
# two-way treatment: scored on this common protocol (does the curriculum still
# do the real task?) AND, as a deliberate follow-up, on its own distribution.
#
# HORIZON 200, not recontact.yaml's 100. v31 measured the conjunction failing
# inside a horizon sized for ONE fingertip; two must be placed. A task key, so
# it is pinned identically across all three gamma arms.
#
# PURE SPARSE ON TWO OF THREE, and it is expected to be hard, not easy: HER is
# the only thing giving a sparse conjunction any gradient. gamma_free pure
# sparse is the honest floor for "can Eq 13 be acquired from free space at all".
#
# SCORE EACH GAMMA CLASS SEPARATELY when reading this. Pooling push/pivot/pinch
# hides which one fails, and the xi interface one-hot (kept on purpose, so the
# state+xi head is identical across templates) is what makes per-class
# conditioning possible in the first place.
#
# NO PREREGISTERED SUCCESS BAR, on purpose. There is no Gamma floor on disk --
# every prior number came from the broken arrival test -- so this sweep's job is
# to MEASURE the floor and the ceiling, not to clear a bar invented now. The
# untrained floor for each protocol is regenerated before scoring
# (docs/TODO.md, and CLAUDE.md's "regenerate the untrained floor for every
# protocol change").
# ===========================================================================

set -e

GROUP="${GROUP:-gamma}"
# An ARRAY, so an empty one expands to nothing rather than to an empty argument
# Hydra would reject. Only the shaped arm fills it.
GUARD_ARG=()
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( i % 3 ))]}

# Hydra list overrides go through shell variables: bash treats [a,b,c] as a
# glob character class and [{a,b}] as brace expansion, so either can reach
# Hydra mangled (CLAUDE.md, Environment).
GMODES="goal_gamma_modes=[push,pivot,pinch]"
INIT_FREE="init_gamma_modes=[free]"
INIT_ALL="init_gamma_modes=[free,push,pivot,pinch]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
COMMON="use_her=true guard_terminates=true her_n_sampled_goal=4 \
learning_starts=10000 eps_v_cm_s=0.5 eps_omega_deg_s=5.0 \
obs_version=2 rich_obs=true normalize_goal_keys=true"
SPARSE="w_d=0 w_a=0 w_F=0 w_m=0 w_T=0"
# P4's terms. w_hold=0.005, NOT 0.02: measured at 0.02 on a 200-tick horizon a
# 2k-step run reached ep_rew_mean 4.57 with success_rate 0.000 and every episode
# at full horizon -- "hold and stall" was already worth 46% of the arrival
# bonus. Every per-tick term is credit-metered for the same reason.
#
# w_prog, NOT w_arrive_pos, and this is a HER correctness constraint rather than
# a preference. w_arrive_pos is metered once per EPISODE in step(); a relabeled
# transition arrives alone, so compute_reward would pay it per tick -- measured
# 3.0/(1-0.99) = 300 of implied Q against goal_reward=10, on ~80% of every
# batch. train_contact now refuses w_arrive_pos alongside use_her. w_prog is
# potential-based, so it is policy-invariant per goal (Ng et al. 1999) and
# relabels EXACTLY, which is the property that matters here.
#
# 0.1: the telescoped total over a full approach is w_prog * d0, and gamma's
# reset distance measures a 15.8cm median / 24.4cm max, so this is ~1.6-2.4 --
# a real gradient at 16-24% of the arrival bonus, not a second objective.
# INFERRED from that arithmetic, not tuned.
#
# WHAT HER DROPS from this arm, bounded and on purpose (reward.RELABEL_DROPPED):
# w_T*horizon = 2.0, hold_cap = 2.0, settle_cap = 0.5, w_guard one latched
# charge. So the relabeled reward is within ~4.5 of the rollout reward against a
# 10.0 bonus. goal_reward and w_prog are reconstructed exactly.
DENSE="w_d=0 w_a=0 w_F=0 w_m=0 w_T=0.01 w_hold=0.005 hold_cap=2.0 \
w_settle=0.02 settle_cap=0.5 settle_radius_cm=1.2 w_prog=0.1"
# Per-outcome guard penalties, SHAPED ARM ONLY. Setting w_guard at all makes
# RewardWeights.dense() true, and train_contact then refuses target_clip -- so
# handing this to a pure-sparse arm would kill the run at startup. Every entry
# stays well below goal_reward=10: at w_m=50/100 push learned to park the object
# where the guard could never fire again (docs/PROGRESS.md v16).
GUARDW="w_guard={object_disturbed: 2.0, overshoot: 2.0, off_board: 5.0}"

case "${GROUP}" in
  base)
    ARMS=(base_v2)
    ARM=${ARMS[0]}
    TOTAL_STEPS=1000000
    HORIZON=100
    # 6.0 on purpose -- see the header. This is what recon_base trained on.
    TASK="gamma_goal=false mask_inactive_finger=true guard_object_still=false \
her_valid_filter=false horizon=${HORIZON} angular_drag_arm_cm=6.0"
    # recontact.yaml's own shaping, kept: recontact has never been pure sparse.
    REWARD="w_T=0.02 w_a=0.01 w_m=2.0"
    CLIP="target_clip=10"
    PINS_LINE="${COMMON} ${TASK} ${REWARD} ${CLIP}"
    ;;
  gamma)
    ARMS=(gamma_free gamma_init gamma_init_noclip gamma_init_shaped)
    ARM=${ARMS[$(( i / 3 ))]}
    TOTAL_STEPS=1000000
    HORIZON=200
    TASK="gamma_goal=true continuous_gamma=true gamma_min_sep_cm=2.0 \
${GMODES} guard_object_still=true her_valid_filter=true \
mask_inactive_finger=false horizon=${HORIZON} angular_drag_arm_cm=3.12"
    INIT="${INIT_FREE}"
    REWARD="${SPARSE}"
    CLIP="target_clip=10"
    case "${ARM}" in
      gamma_free)        ;;
      gamma_init)        INIT="${INIT_ALL}" ;;
      # THE CONTROL FOR target_clip. Shaping FORCES target_clip=null (the
      # [0, clip] bound assumes Q* <= goal_reward, which a per-tick penalty
      # breaks), so gamma_init_shaped moves two things at once and target_clip
      # is the one that took recontact from 1/6 to 6/6 seeds. Without this arm
      # a failed shaped arm is uninterpretable in exactly the way v33 was:
      # three arms cannot bisect two changes. Costs 3 cells, buys the whole
      # reading -- gamma_init - gamma_init_noclip is the price of the clamp,
      # gamma_init_shaped - gamma_init_noclip is the price of the shaping.
      gamma_init_noclip) INIT="${INIT_ALL}"; CLIP="target_clip=null" ;;
      gamma_init_shaped) INIT="${INIT_ALL}"; REWARD="${DENSE}"; CLIP="target_clip=null"
                         GUARD_ARG=("${GUARDW}") ;;
    esac
    # The COMMON protocol is init_gamma_modes=[free]: the canonical, harder
    # task. Arms that train on the loosened start are scored here AND, as a
    # deliberate second step, on their own distribution.
    PINS_LINE="${COMMON} ${TASK} ${INIT_FREE} ${SPARSE} target_clip=10"
    ;;
  *) echo "unknown GROUP ${GROUP} (want base|gamma)" >&2; exit 2 ;;
esac

TEMPLATE="recontact"
CKPT_FREQ=250000
EVAL_EPS=32
EXTRA_OVERRIDE="${COMMON} ${TASK} ${INIT} ${REWARD} ${CLIP} \
ckpt_freq=${CKPT_FREQ} diag_eval_episodes=${EVAL_EPS}"
RUN_TAG="recon_${ARM}_s${SEED}"

SWEEP_DIR="logs/sweep_${SLURM_ARRAY_JOB_ID}"
mkdir -p "${SWEEP_DIR}"

# THE BENCHMARK PROTOCOL, written next to the runs rather than retyped in the
# scorer -- a protocol that lives only in a launcher comment is what made every
# v25 cross-version comparison wrong, and a hardcoded portal in the scorer is
# what invalidated v33's first scoring run. Written via mv, which is atomic:
# array tasks race here with identical content.
cat > "${SWEEP_DIR}/.PINS.$$" <<PINSEOF
${PINS_LINE}
PINSEOF
mv -f "${SWEEP_DIR}/.PINS.$$" "${SWEEP_DIR}/PINS.txt"

RUN_ID="$(date +%Y%m%d_%H%M%S)_jobid${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${RUN_TAG}"
RUN_DIR="${SWEEP_DIR}/${RUN_ID}"; mkdir -p "${RUN_DIR}"

exec >  "${RUN_DIR}/run.out"; exec 2> "${RUN_DIR}/run.err"
{ echo "RUN_ID=${RUN_ID}"; echo "HOST=$(hostname)"; echo "DATE=$(date -Iseconds)";
  echo "GROUP=${GROUP} ARM=${ARM}";
  echo "TEMPLATE=${TEMPLATE} TOTAL_STEPS=${TOTAL_STEPS} EXTRA_OVERRIDE=${EXTRA_OVERRIDE}";
  echo "GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null)";
  # GIT_COMMIT alone could not tell job 40944664 (pre-fix) from 40957220
  # (post-fix): both recorded a7c153a because the fix was uncommitted. Record
  # whether the tree was dirty and a hash of the diff, so a run's provenance is
  # checkable rather than a guess (docs/PROGRESS.md, v20).
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
  ${EXTRA_OVERRIDE} "${GUARD_ARG[@]}" \
  out_dir="${RUN_DIR}" \
  wandb=true wandb_run_name="${RUN_TAG}_${SLURM_ARRAY_JOB_ID}" \
  wandb_group="reconD_${SLURM_ARRAY_JOB_ID}" || rc=$?
# `|| rc=$?` and NOT a bare `rc=$?`: `set -e` is on, so a training
# failure would exit the task RIGHT HERE -- skipping the last-task-standing
# block below, which is what moves the staging logs and submits
# finalize.sh. A sweep whose last cell dies would then silently produce no
# scoring and leave its logs orphaned, which is how 621 staging files
# accumulated. rc is initialised so the trap still reports success.

# Am I the last task standing? `-o "%i"`, NOT "%A_%a": on this Slurm %a renders
# as the ACCOUNT name ("43892866_hankyang_lab"), so the grep matched nothing,
# `still` never reached 0, and this block silently never ran on v29, v32 or v33
# -- verified by three sweeps with no slurm_logs/ and 621 orphaned staging
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
  # finalize reads the template from meta.txt, so this needs no extra argument.
  sbatch slurm/finalize.sh "${SWEEP_DIR}" "reconD_${GROUP}"
fi
exit $rc
