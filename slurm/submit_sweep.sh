#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=overlap_donepatch
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# SUPERSEDED by slurm/sweep_push_cone.sh and slurm/sweep_recontact_clip.sh
# (v21). Kept as the record of job 41613939. Do not submit this one.
# ===========================================================================
# v20 sweep. Gate-verified 4/4 plus the correctness checks recorded in
# docs/PROGRESS.md's v20 entry before submission.
#
# ---------------------------------------------------------------------------
# Tasks 0-3: PUSH, 150k. A 2x2: same_room_goal_prob {0.0, 1.0} x seed {0,1}.
# ---------------------------------------------------------------------------
# Baseline for comparison is job 40957220 cells 2-3 (push_nosettled), which
# shared require_settled=false with every cell here.
#
# Constant across all four, and the delta vs that baseline:
#   min_progress_cm=0.5  ->  min_progress_ticks=1  (cm filter removed entirely)
#
#   The distance filter was coupled to arrival_eps by the triangle inequality:
#   `arrived` needs |ag_{t+1} - dg| < arrival_eps and the filter needs
#   |dg - ag_t| > min_progress_cm, and ag_{t+1} ~= ag_t for a slow object, so
#   both hold only if the object moved > (min_progress_cm - arrival_eps) in
#   that ONE tick. Measured median per-tick motion is 0.0098cm against a
#   required 0.1cm, so the filter kept 0.33-1.0% of HER pairs and the survivors
#   were 21-43x median tick speed -- i.e. the fastest-moving-object moments,
#   which under a settled success criterion are the least like a real success.
#
#   ticks=1 (not 3 or 10) because that is the reference HER convention, not an
#   invention: OpenAI baselines computes `future_t = t_samples + 1 + offset`,
#   strictly future. SB3 deviates ("our implementation is inclusive: current
#   transition can be sampled"), and v20 measured the cost -- 745 pairs at
#   lag 0, 100% of which score a reward tautologically because the goal IS that
#   transition's own outcome. ticks=1 deletes exactly those and keeps 89.8% of
#   the remaining positives. Larger thresholds are unvalidated and are NOT swept
#   here; they also go blind if episodes ever lengthen (at guard_terminates=false,
#   lag>=10 passes 88% of positives whose object path length is exactly 0.000cm).
#
# The swept axis, and why it is the one that matters:
#   same_room_goal_prob 0.0 -> 1.0 shortens the initial object-to-goal distance.
#   Measured over 600 resets:
#       srg=0.0 : min 12.96  p10 17.82  median 25.10 cm
#       srg=1.0 : min  0.27  p10  2.90  median  7.04 cm
#   At srg=0.0 NOT ONE episode in 600 starts closer than 12.96cm, while HER's
#   positive examples are capped at arrival_eps + one tick of object motion
#   (0.81cm measured, 0.85cm predicted) and real success needs 0.4cm. That is
#   ZERO overlap between what HER can teach and what success requires -- which
#   is the standing explanation for why no push cell has ever succeeded under
#   any filter, reward weight, or budget. At srg=1.0 the p10 is 2.90cm and 0.3%
#   of episodes start already inside arrival_eps, so the REAL reward can fire
#   without HER at all.
#   srg=0.0 is retained as the internal control, so the filter change is
#   measured against the old setting and the overlap change is measured on top
#   of it. This is a stand-in for memo Eq 15's curriculum (which expands the
#   initiation set backward from the target and is still unimplemented -- see
#   docs/TODO.md); it changes the task to a within-room push rather than
#   shrinking the difficulty axis of the cross-room one.
#
# PREREGISTERED PREDICTIONS (recorded so they can be checked against, per the
# discipline in docs/PROGRESS.md; the second row is the informative one):
#   - critic_loss stays bounded in all four but settles ABOVE the 0.67 dead
#     floor of job 40957220's s0 cells, because positives go from 1.0% to 60.6%
#     of pairs. If it diverges again, the done-flag patch is not sufficient and
#     there is a second value bug.
#   - the "dead fixed point" (critic->0.67, ent_coef->0.002, ep_len declining)
#     DISAPPEARS. If it persists with ~60x the positive data, the bottleneck is
#     the actor / exploration, not the critic's inputs. That distinction is the
#     main thing this arm buys.
#   - srg=0.0 cells: real success stays 0.0, because the zero-overlap above is
#     untouched by the filter change.
#   - srg=1.0 cells: FIRST verified push success expected here, if anywhere.
#     If these also stay at 0.0, the blocker is contact retention rather than
#     overlap -- v17 measured a scripted closed-loop rule holding contact 30/30
#     against the trained policy's median break at tick 6, so that is the next
#     place to look.
#
# ---------------------------------------------------------------------------
# Tasks 4-9: RECONTACT, 300k, 6 seeds, guard_terminates left at its default.
# ---------------------------------------------------------------------------
# Baseline for comparison is job 40910275 cells 2-5.
#   Delta 1: recontact now uses DonePatchedHerReplayBuffer, so a relabeled
#   transition scoring the arrival bonus is marked terminal. v19 applied this
#   only to push's subclass; v20 measured the cost of the omission -- Q(s0)
#   +44.8/+40.3 against a realized return of ~-1.5 at 1M, with the policy
#   parking the finger 0.77-1.89cm short while holding the object at exactly
#   0.000cm/s. Verified before submission: 14/14 positive-reward virtual
#   samples now come back done=1, where the stock buffer gave 0/14.
#   Delta 2: budget 1M -> 300k. Peak held-out eval lands at 120-360k in every
#   prior cell and the 700k trough is 4.6x below it, so ~800k of every 1M cell
#   was spent worse than its own 200k checkpoint. The freed wall-clock buys
#   seeds instead, which is what the seed effect -- now reproduced on three
#   independent axes -- actually needs.
#   Delta 3: guard_terminates is no longer swept. gttrue/gtfalse trajectories
#   nearly overlap at both seeds; the axis is spent.
#
# NOT changed here, deliberately: physics.obs()'s rel_target slot (and with it
# push's bespoke HER buffer) is redundant -- it equals
# (desired_goal - achieved_goal)/pos_scale, a linear function of two arrays the
# policy already receives, and its only consumers in the repo are the patch that
# repairs it. Removing it would invalidate every existing push checkpoint, so it
# is held out of this sweep. See docs/TODO.md.
# ===========================================================================

i=$SLURM_ARRAY_TASK_ID
if [ "$i" -lt 4 ]; then
  TEMPLATE="push"
  SEED=$(( i % 2 ))
  TOTAL_STEPS=150000
  # 0,1 -> srg 0.0 (internal control, the old setting); 2,3 -> srg 1.0.
  SRG=$([ "$i" -lt 2 ] && echo 0.0 || echo 1.0)
  # min_progress_cm is deliberately NOT passed: it defaults to null, which
  # disables the distance filter entirely. min_progress_ticks alone gates.
  EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true board_w_cm=50.0 board_h_cm=30.0 portals=[{x:25.0,y_lo:5.0,y_hi:25.0}] min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 same_room_goal_prob=${SRG} require_settled=false"
  RUN_TAG="push_srg${SRG}_ticks1_s${SEED}"
else
  TEMPLATE="recontact"
  SEED=$(( i - 4 ))
  TOTAL_STEPS=300000
  EXTRA_OVERRIDE="use_her=true w_T=0.0 w_a=0.0 w_m=0.0 guard_terminates=true"
  RUN_TAG="recontact_donepatch_s${SEED}"
fi

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
  wandb_group="overlap_donepatch_${SLURM_ARRAY_JOB_ID}"
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
