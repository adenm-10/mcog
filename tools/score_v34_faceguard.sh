#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=v34guard
#SBATCH --output=logs/slurm_staging/%j_v34guard.out
#SBATCH --error=logs/slurm_staging/%j_v34guard.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-03:00:00

# WHAT DOES PUSH'S 0.826 COST WHEN THE EDGE LABEL IS ENFORCED?
#
#   sbatch tools/score_v34_faceguard.sh [sweep] [ckpts]
#
# Eq 40 makes the contacted face an edge parameter, so a finger that walks to
# another face has violated the option it was told to execute. Sweep A's
# headline was measured with guard_face=false.
#
# Two measurements say this is not a rounding error:
#
#   v32 (logs/eval/v32_faceprobe): both best policies left the contacted face
#   on 82-87% of episodes, and the ARM ORDERING INVERTED under the guard --
#   curric 0.683 -> 0.483 against base 0.600 -> 0.583. So the guarded number
#   is not a constant offset and cannot be inferred from the unguarded one.
#
#   tools/probe_reachable.py, 2026-09-04: with guard_face=adjacent, `wrong_face`
#   becomes the single largest exit (434 of 1024 open-loop rollouts) and the
#   fraction of reachable displacements landing 120-180deg BEHIND the contact
#   collapses from 0.176 to 0.014. Most "behind" reachability was bought by
#   leaving the face.
#
# guard_face is a TASK key (it changes termination), so each setting is its own
# digest and its own experiment -- these are absolute columns, read as a DELTA
# against the sweep's own unguarded numbers, not as one benchmark. A frozen
# checkpoint is licensed here because the policy's outputs do not depend on the
# guard (CLAUDE.md: valid for changes the policy's outputs don't depend on).
#
# `adjacent` and `strict` only. `false` is already on disk -- it IS the v34
# scoring -- and re-running it would only add a way for the two to disagree.
set -e

SWEEP="${1:-logs/sweep_44180162}"
CKPTS="${2:-model.zip model_best.zip}"
BASE="$(cat "${SWEEP}/PINS.txt")"

case "${BASE}" in
  *guard_face=false*) ;;
  *) echo "FATAL: ${SWEEP}/PINS.txt does not pin guard_face=false, so the" >&2
     echo "       flip below would not be a single-factor change." >&2
     exit 1 ;;
esac

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu

for MODE in adjacent true; do
  PINS="${BASE/guard_face=false/guard_face=${MODE}}"
  case "${PINS}" in
    *"guard_face=${MODE}"*) ;;
    *) echo "FATAL: flip to guard_face=${MODE} did not apply" >&2; exit 1 ;;
  esac
  OUT="logs/eval/v34_faceguard_${MODE}"
  python tools/score_sweep.py "${SWEEP}" \
    --out-dir "${OUT}" --template push --jobs 4 \
    --ckpt ${CKPTS} --pins "${PINS}"

  # The protocol lives beside the numbers, never in a launcher comment.
  {
    echo "# Sweep A scored with guard_face=${MODE}"
    echo
    echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/score_v34_faceguard.sh\`."
    echo "Cells: ${SWEEP}. Checkpoints: ${CKPTS}."
    echo
    echo "TASK pins = ${SWEEP}/PINS.txt with guard_face flipped from false to"
    echo "\`${MODE}\` and nothing else touched:"
    echo
    echo '```'
    echo "${PINS}"
    echo '```'
    echo
    echo "guard_face is a TASK key, so this is its OWN digest and its own"
    echo "experiment. Read it as a DELTA against the sweep's unguarded numbers"
    echo "(logs/eval/sweepA_*), never pooled with them."
    echo
    echo "## Digests"
    echo '```'
    python - "${OUT}" <<'PY'
import glob, json, os, sys
seen = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    seen.setdefault(json.load(open(p))["env_digest"], []).append(os.path.basename(p))
for d, ps in seen.items():
    print(f"{d}: {len(ps)} evals")
PY
    echo '```'
  } > "${OUT}/PROTOCOL.md"
done

echo "done: logs/eval/v34_faceguard_{adjacent,true}"
