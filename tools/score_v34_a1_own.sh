#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=a1own
#SBATCH --output=logs/slurm_staging/%j_a1own.out
#SBATCH --error=logs/slurm_staging/%j_a1own.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-03:00:00

# THE SECOND HALF OF SWEEP A'S TWO-WAY TREATMENT.
#
# finalize.sh scores all nine Sweep A cells on the sweep's shared PINS.txt,
# which is the ALONG-FACE spawn (push_spawn_along_frac=0.7, digest
# 646ba4ae1fd4). That answers "does a centre-trained policy do the randomized
# task?" but it makes A1's number a TRANSFER number rather than its own.
#
# This run flips that one key to null and nothing else, so the whole sweep is
# also scored on the FACE-CENTRE protocol -- digest 249434216cd2, which is
# v32's and v33's. Two things come out of it:
#
#   A1 here is A1's OWN task, and directly comparable to every archived
#   v32/v33 figure (ctl 0.674, curric 0.674, the six v33 scaffold prices).
#   That comparison is the only thing that can tell us whether v33's prices
#   were read at an unconverged budget -- v33 ran 600k, this ran 1.2M.
#
#   A2/A3 here are the reverse transfer: "does an along-face-trained policy
#   still do the axis-aligned task?" Together with finalize.sh's numbers that
#   is the full 3 arms x 2 protocols table, and the DELTA is the quantity of
#   interest, not either column alone.
#
# push_spawn_along_frac is a TASK key, so the two protocols are two digests and
# two experiments (CLAUDE.md). Pins are derived from the sweep's own PINS.txt
# by sed rather than retyped, because a protocol duplicated between a launcher
# and a scorer is how every v25 cross-version comparison came out wrong.
set -e

SWEEP="${1:-logs/sweep_44180162}"
OUT="${2:-logs/eval/sweepA_a1_own}"

PINS="$(sed 's/push_spawn_along_frac=0\.7/push_spawn_along_frac=null/' \
        "${SWEEP}/PINS.txt")"
case "${PINS}" in
  *push_spawn_along_frac=null*) ;;
  *) echo "FATAL: the spawn flip did not apply; ${SWEEP}/PINS.txt changed" >&2
     exit 1 ;;
esac

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu

python tools/score_sweep.py "${SWEEP}" \
  --out-dir "${OUT}" --template push --jobs 4 \
  --ckpt model.zip model_best.zip --pins "${PINS}"

# The protocol lives beside the numbers, never in a launcher comment.
{
  echo "# Sweep A on the FACE-CENTRE protocol (A1's own task)"
  echo
  echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/score_v34_a1_own.sh\`."
  echo "Cells: ${SWEEP}. Expected digest 249434216cd2 (v32/v33's own)."
  echo
  echo "TASK pins, = ${SWEEP}/PINS.txt with push_spawn_along_frac flipped to"
  echo "null and nothing else touched:"
  echo
  echo '```'
  echo "${PINS}"
  echo '```'
  echo
  echo "INTERFACE keys are read per cell from its own meta.txt (obs v1 for"
  echo "a1/a2, obs v2 + normalized goal keys for a3), so they are outside the"
  echo "digest and all nine cells share one benchmark."
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

echo "done: ${OUT}"
