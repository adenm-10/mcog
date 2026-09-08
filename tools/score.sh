#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=score
#SBATCH --output=logs/slurm_staging/%j_score.out
#SBATCH --error=logs/slurm_staging/%j_score.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0-08:00:00

# ONE SCORER. Every protocol a sweep needs, as a table of variants.
#
#   sbatch tools/score.sh <sweep_dir> [variant ...]     # default: all applicable
#
# Replaces score_rungs.sh, score_v35_rungs.sh, score_v34_a1_own.sh and
# score_v34_faceguard.sh. The last two are deleted; the first two survive ONLY
# until Sweeps C and D land -- their running tasks already hold
# RUNG_SCORER=tools/score_rungs.sh in memory and will sbatch that exact path
# when the last cell exits. Delete both once those sweeps have scored.
#
# All four were copies of the same four steps:
# read PINS.txt -> transform one key -> ASSERT the transform applied -> call
# score_sweep.py. Only the key differed. A fifth copy was written for every new
# question; this is the table instead.
#
# | variant    | applies when                        | pins                  | ckpts    |
# |------------|-------------------------------------|-----------------------|----------|
# | rungs      | model_*_steps.zip on disk           | unchanged             | discovered|
# | countguard | a cell ran xi_gamma_mode=count      | + guard_contact_count=1| final+best|
# | faceguard  | pins carry guard_face=false         | -> adjacent           | final    |
# | settled    | pins carry require_settled=false    | -> true               | final+best|
# | spawn      | pins carry push_spawn_along_frac=0.7| -> null               | final+best|
# | perarm     | PINS.<arm>.txt exists               | that file, per arm    | final+best|
#
# WHY EVERY TRANSFORM IS ASSERTED. A sed that silently misses scores the sweep
# on the protocol it was trying to change and the numbers look fine. That is how
# v33 was scored on a 20cm doorway instead of its own 10cm one -- 36 of 60
# benchmark episodes differed and only the digest gave it away.
#
# WHY PERARM EXISTS. Sweep D's two arms train on DIFFERENT tasks, so there is no
# single common protocol and finalize.sh (which assumes one) is deliberately not
# wired to it. Before this variant that meant scoring D by hand.
set -e

SWEEP="${1:?usage: sbatch tools/score.sh <sweep_dir> [variant ...]}"; shift
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
TAG="$(basename "${SWEEP}" | sed 's/^sweep_//')"

source ~/.bashrc; module load python; mamba activate tsmc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export JAX_PLATFORMS=cpu
export MPLBACKEND=Agg

PINS="$(cat "${SWEEP}/PINS.txt")"
WANT="$*"
want () { [ -z "${WANT}" ] || case " ${WANT} " in *" $1 "*) return 0;; *) return 1;; esac; }

# from, to -- one key, and the substitution is ASSERTED, never assumed.
flip () {
  local OUT; OUT="$(printf '%s' "${PINS}" | sed "s/$1/$2/")"
  case "${OUT}" in
    *"$2"*) printf '%s' "${OUT}" ;;
    *) echo "FATAL: flip '$1' -> '$2' did not apply; ${SWEEP}/PINS.txt changed" >&2
       exit 1 ;;
  esac
}

# The protocol lives BESIDE the numbers, never only in a launcher comment.
protocol_md () {   # dir, pins, note
  { echo "# $(basename "$1")"; echo
    echo "Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/score.sh\`."
    echo "Cells: ${SWEEP}. $3"; echo; echo '```'; echo "$2"; echo '```'; echo
    echo "INTERFACE keys are read per cell from its own meta.txt, so they sit"
    echo "outside the digest and every cell shares one benchmark."; echo
    echo "## Digests"; echo '```'
    python - "$1" <<'PY'
import glob, json, os, sys
seen = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    seen.setdefault(json.load(open(p))["env_digest"], []).append(p)
for d, ps in seen.items():
    print(f"{d}: {len(ps)} evals")
PY
    echo '```'; } > "$1/PROTOCOL.md"
}

score () {   # out-suffix, pins, note, ckpt...
  local SUF="$1" P="$2" NOTE="$3"; shift 3
  local DIR="logs/eval/${TAG}_${SUF}"
  echo "=== ${SUF} ==="
  python tools/score_sweep.py "${SWEEP}" --out-dir "${DIR}" --jobs 4 \
    --ckpt "$@" --pins "${P}"
  protocol_md "${DIR}" "${P}" "${NOTE}"
}

# --- rungs: the budget axis. DISCOVERED from disk, because a cell killed at the
# wall still yields whatever snapshots it wrote. Sweep B's arm ordering flipped
# between 1.8M and 2.4M, so one budget can rank arms backwards.
if want rungs; then
  RUNGS=$(ls "${SWEEP}"/*/model_*_steps.zip 2>/dev/null \
          | xargs -rn1 basename | sort -u -t_ -k2 -n)
  if [ -n "${RUNGS}" ]; then
    score rungs "${PINS}" "Pins = PINS.txt verbatim; the budget axis." ${RUNGS}
  else
    echo "no intermediate checkpoints under ${SWEEP} -- was ckpt_freq set?"
  fi
fi

# --- countguard: enforcing a count is a different TASK from being told one, so
# `count` carries its own digest. The WHOLE sweep is scored with the guard on,
# because the quantity of interest is the paired DELTA against ctl.
if want countguard && grep -qs "xi_gamma_mode=count" "${SWEEP}"/*/meta.txt; then
  score countguard "${PINS} guard_contact_count=1" \
        "Pins = PINS.txt + guard_contact_count=1 (a TASK key)." \
        model.zip model_best.zip
fi

# --- faceguard / settled / spawn: zero-shot re-scores, valid on a frozen
# checkpoint because each changes the arrival TEST or the reset sampler, not the
# observation or the action space. Read as a DELTA, never pooled.
if want faceguard && case "${PINS}" in *guard_face=false*) true;; *) false;; esac; then
  score faceguard "$(flip 'guard_face=false' 'guard_face=adjacent')" \
        "Pins = PINS.txt with guard_face -> adjacent. Zero-shot: a DELTA." model.zip
fi
if want settled && case "${PINS}" in *require_settled=false*) true;; *) false;; esac; then
  score settled "$(flip 'require_settled=false' 'require_settled=true')" \
        "Pins = PINS.txt with require_settled -> true. Zero-shot: a DELTA." \
        model.zip model_best.zip
fi
if want spawn && case "${PINS}" in *push_spawn_along_frac=0.7*) true;; *) false;; esac; then
  score spawn "$(flip 'push_spawn_along_frac=0\.7' 'push_spawn_along_frac=null')" \
        "Pins = PINS.txt with the along-face spawn -> face centre." \
        model.zip model_best.zip
fi

# --- perarm: arms that trained on DIFFERENT tasks. Each gets its own protocol
# and its own digest, and they are never pooled.
if want perarm; then
  for PF in "${SWEEP}"/PINS.*.txt; do
    [ -e "${PF}" ] || continue
    ARM="$(basename "${PF}" .txt)"; ARM="${ARM#PINS.}"
    echo "=== perarm: ${ARM} ==="
    python tools/score_sweep.py "${SWEEP}" --out-dir "logs/eval/${TAG}_${ARM}" \
      --jobs 4 --arm "${ARM}" --ckpt model.zip model_best.zip \
      --pins "$(cat "${PF}")"
    protocol_md "logs/eval/${TAG}_${ARM}" "$(cat "${PF}")" \
      "Arm ${ARM} only, on its OWN protocol. Not comparable to another arm."
  done
fi

# --- the split read: two predicates, three strata, never pooled.
for D in logs/eval/${TAG} logs/eval/${TAG}_*; do
  [ -d "$D" ] && { echo; echo "### $D"; python tools/split_floor.py "$D" || true; }
done
echo "done: logs/eval/${TAG}_*"
