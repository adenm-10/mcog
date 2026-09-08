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
  # whether the tree was dirty and a hash of the diff, so provenance is
  # checkable rather than a guess (docs/PROGRESS.md, v20).
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
  wandb_group="${GROUP:-lean}_${SLURM_ARRAY_JOB_ID}"
rc=$?

# AM I THE LAST TASK STANDING? `-o "%i"`, NOT "%A_%a": on this Slurm %a renders
# as the ACCOUNT name ("43892866_hankyang_lab"), so the grep matched nothing,
# `still` never reached 0, and this block silently never ran -- on v29, v32 or
# v33, leaving 621 orphaned staging files. submit_sweep.sh fixed its own INLINE
# copy on 2026-09-04; this shared one was still broken until 2026-09-08, which is
# the same "one behaviour, two copies" defect the fix was about. One copy now.
still=$(squeue -h -j "${SLURM_ARRAY_JOB_ID}" -t PENDING,RUNNING -o "%i" \
        | grep -v "^${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}$" | wc -l)
# mkdir is atomic, so two tasks finishing in the same instant cannot both submit.
if [ "${still}" -eq 0 ] && mkdir "${SWEEP_DIR}/.finalized" 2>/dev/null; then
  mkdir -p "${SWEEP_DIR}/slurm_logs"
  find logs/slurm_staging -maxdepth 1 -type f \
       \( -name "${SLURM_ARRAY_JOB_ID}_*.out" -o -name "${SLURM_ARRAY_JOB_ID}_*.err" \) \
       -exec mv -t "${SWEEP_DIR}/slurm_logs/" {} + 2>/dev/null
  # Score + render as a FOLLOW-ON job, not inline: this cell may be near its own
  # wall, and scoring 24-60 checkpoints is not free. Only when the launcher asks
  # for it, so the older launchers that source this file keep their behaviour.
  if [ -n "${FINALIZE_TAG:-}" ]; then
    sbatch slurm/finalize.sh "${SWEEP_DIR}" "${FINALIZE_TAG}"
    [ -n "${RUNG_SCORER:-}" ] && sbatch "${RUNG_SCORER}" "${SWEEP_DIR}"
  fi
fi
exit $rc
