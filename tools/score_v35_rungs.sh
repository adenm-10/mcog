#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=v35rungs
#SBATCH --output=logs/slurm_staging/%j_v35rungs.out
#SBATCH --error=logs/slurm_staging/%j_v35rungs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-06:00:00

# SWEEP B'S BUDGET AXIS, AND THE TWO LOOSENED-TASK PROTOCOLS.
#
#   sbatch tools/score_v35_rungs.sh <sweep_dir>
#
# finalize.sh scores model.zip and model_best.zip on the common protocol. That
# is one rung of four, and the budget axis IS the sweep, so this scores the
# three intermediate snapshots too. Rungs: 600k (v33's budget), 1.2M (Sweep A's),
# 1.8M, and 2.4M = model.zip from finalize.sh.
#
# It also does the second half of the two-way treatment for the two arms that
# train under a loosened TASK key. Each loosened protocol scores the WHOLE sweep,
# not just its own arm, because the interesting quantity is the DELTA between the
# loosened arm and ctl ON the loosened task -- v33's faceguard arm is why: the
# policy that never saw the constraint scored HIGHER under it (+0.062), so the
# tight-benchmark reading alone had the sign of the conclusion wrong.
#
# Pins come from the sweep's own PINS.txt with one key flipped by sed and the
# flip asserted, never retyped.
set -e

SWEEP="${1:?usage: sbatch tools/score_v35_rungs.sh <sweep_dir>}"
PINS="$(cat "${SWEEP}/PINS.txt")"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu

flip () {   # from, to -- one key, asserted
  local OUT
  OUT="$(printf '%s' "${PINS}" | sed "s/$1/$2/")"
  case "${OUT}" in
    *"$2"*) printf '%s' "${OUT}" ;;
    *) echo "FATAL: flip $1 -> $2 did not apply; ${SWEEP}/PINS.txt changed" >&2
       exit 1 ;;
  esac
}

# ---- the budget axis, common protocol (expect 249434216cd2)
python tools/score_sweep.py "${SWEEP}" \
  --out-dir logs/eval/sweepB_rungs --template push --jobs 4 \
  --ckpt model_600000_steps.zip model_1200000_steps.zip model_1800000_steps.zip \
  --pins "${PINS}"

# ---- widecone's own distribution
WIDE="$(flip 'push_cone_deg=30' 'push_cone_deg=90')"
python tools/score_sweep.py "${SWEEP}" \
  --out-dir logs/eval/sweepB_widecone_own --template push --jobs 4 \
  --ckpt model.zip model_best.zip --pins "${WIDE}"

# ---- spread's own distribution
SPRD="$(flip 'object_theta_spread_deg=null' 'object_theta_spread_deg=90')"
python tools/score_sweep.py "${SWEEP}" \
  --out-dir logs/eval/sweepB_spread_own --template push --jobs 4 \
  --ckpt model.zip model_best.zip --pins "${SPRD}"

# The protocol lives beside the numbers, never in a launcher comment.
for pair in "sweepB_rungs:${PINS}" "sweepB_widecone_own:${WIDE}" \
            "sweepB_spread_own:${SPRD}"; do
  DIR="logs/eval/${pair%%:*}"
  {
    echo "# ${pair%%:*}"
    echo
    echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/score_v35_rungs.sh\`."
    echo "Cells: ${SWEEP}. Pins = its PINS.txt with at most one key flipped."
    echo
    echo '```'
    echo "${pair#*:}"
    echo '```'
    echo
    echo "INTERFACE keys are read per cell from its own meta.txt, so they sit"
    echo "outside the digest and every cell shares one benchmark."
    echo
    echo "## Digests"
    echo '```'
    python - "${DIR}" <<'PY'
import glob, json, os, sys
seen = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    seen.setdefault(json.load(open(p))["env_digest"], []).append(os.path.basename(p))
for d, ps in seen.items():
    print(f"{d}: {len(ps)} evals")
PY
    echo '```'
  } > "${DIR}/PROTOCOL.md"
done

echo "done: logs/eval/sweepB_{rungs,widecone_own,spread_own}"
