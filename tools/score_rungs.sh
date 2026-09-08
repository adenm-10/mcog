#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=rungs
#SBATCH --output=logs/slurm_staging/%j_rungs.out
#SBATCH --error=logs/slurm_staging/%j_rungs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-06:00:00

# THE BUDGET AXIS, plus the two-way score for any arm that moved a TASK key.
#
#   sbatch tools/score_rungs.sh <sweep_dir>
#
# Generalized from tools/score_v35_rungs.sh, which hardcoded Sweep B's arm names
# and its two loosened protocols. Here the intermediate rungs are DISCOVERED from
# what is on disk and the second protocol is derived from the sweep's own
# PINS.txt, so the same script serves Sweep C and anything after it.
#
# WHY THE RUNGS ARE THE POINT. finalize.sh scores model.zip and model_best.zip:
# that is one budget of four. Sweep B measured the arm ordering FLIPPING between
# 1.8M and 2.4M -- widecone went 2nd -> 3rd -> tied -> 1st -- so a sweep read at
# a single budget can rank arms backwards. Every adoption decision has to hold at
# two adjacent rungs or it is not a decision.
#
# THE TWO-WAY SCORE. An arm that moved a TASK key trains on a different task, so
# it needs scoring on its OWN protocol as well as the common one, and the WHOLE
# sweep is scored on that protocol rather than just that arm -- the quantity of
# interest is the DELTA against ctl on the loosened task. v33's faceguard arm is
# why: the policy that never saw the constraint scored HIGHER under it (+0.062),
# so the common-benchmark reading alone had the sign of the conclusion wrong.
#
# Pins come from the sweep's own PINS.txt, never retyped, with any flip applied
# by sed and then ASSERTED.
set -e

SWEEP="${1:?usage: sbatch tools/score_rungs.sh <sweep_dir>}"
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
TAG="$(basename "${SWEEP}" | sed 's/^sweep_//')"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

PINS="$(cat "${SWEEP}/PINS.txt")"

# --- the intermediate rungs, DISCOVERED rather than hardcoded ---------------
# A cell killed at the wall still yields whatever snapshots it wrote, so the set
# is whatever is actually on disk, not whatever the launcher intended.
RUNGS=$(ls "${SWEEP}"/*/model_*_steps.zip 2>/dev/null \
        | xargs -rn1 basename | sort -u -t_ -k2 -n)
if [ -z "${RUNGS}" ]; then
  echo "no intermediate checkpoints under ${SWEEP} -- was ckpt_freq set?" >&2
  exit 1
fi
echo "rungs found: ${RUNGS}"

python tools/score_sweep.py "${SWEEP}" \
  --out-dir "logs/eval/${TAG}_rungs" --jobs 4 \
  --ckpt ${RUNGS} --pins "${PINS}"

# --- the second protocol, for any arm that moved a task key -----------------
# Sweep C's is `count`: guard_contact_count is a TASK key, so `count` trains on a
# different task from the other three and carries its own digest. Scoring the
# WHOLE sweep with the guard ON is what makes (count - ctl) a paired comparison
# on identical episodes.
COUNT_PINS="${PINS} guard_contact_count=1"
if grep -qs "xi_gamma_mode=count" "${SWEEP}"/*/meta.txt; then
  echo "=== two-way: whole sweep under guard_contact_count=1 ==="
  python tools/score_sweep.py "${SWEEP}" \
    --out-dir "logs/eval/${TAG}_countguard" --jobs 4 \
    --ckpt model.zip model_best.zip --pins "${COUNT_PINS}"
fi

# --- free zero-shot re-scores, D5 ------------------------------------------
# guard_face costs 0.042-0.083 and IS enforceable; it stays off during training
# only so no arm is floored. This prices it on every cell for four minutes.
FACE_PINS="$(echo "${PINS}" | sed 's/guard_face=false/guard_face=adjacent/')"
[ "${FACE_PINS}" = "${PINS}" ] && { echo "guard_face flip did not apply" >&2; exit 2; }
echo "=== zero-shot: guard_face=adjacent ==="
python tools/score_sweep.py "${SWEEP}" \
  --out-dir "logs/eval/${TAG}_faceguard" --jobs 4 \
  --ckpt model.zip --pins "${FACE_PINS}"

# --- the split read: two predicates, three strata ---------------------------
for D in "logs/eval/${TAG}" "logs/eval/${TAG}_rungs" "logs/eval/${TAG}_countguard"; do
  [ -d "$D" ] && { echo; echo "### $D"; python tools/split_floor.py "$D"; }
done

echo "done: logs/eval/${TAG}_rungs and the two-way scores"
