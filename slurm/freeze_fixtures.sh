#!/usr/bin/env bash
# scripts/freeze_fixtures.sh -- freeze tests/fixtures from a 90k run.
#
# Ends Phase A (handoff sec 8.4). Own commit. Runs in parallel with the queued
# acceptance job; nothing here depends on acceptance landing.
#
# NEVER on a login node. salloc first:
#   salloc --partition=gpu_test --nodes=1 --cpus-per-task=8 --mem=16G \
#          --gres=gpu:nvidia_a100_3g.20gb:1 --time=01:00:00
#
# Differences from handoff sec 7, all deliberate:
#   --eval-seed is pinned explicitly. sample_eval_pairs(maze, num_pairs,
#     eval_seed) reads cfg["eval_seed"], NOT cfg["seed"]. The doc's claim that
#     "--eval-episodes 32 --seed 1 fix the pair set" is wrong: --seed 1 sets the
#     TRAINING seed and touches the pair set not at all. Without --eval-seed the
#     fairness anchor is inherited silently from base.yaml.
#   --diag-eval-freq 5000 so PeriodicEvalCallback actually fires inside a
#     10,000-step-per-region budget. This is the FIRST EXECUTION of A3's
#     double-count fix and D2's latch fix (handoff sec 4: source-inspected only).
#     It costs ~3 minutes instead of the 2-hour acceptance job, and if the
#     callback raises on first fire you find out now.
#   --wall-margin 0.0 and --horizon 200/600 are KEPT WRONG on purpose. Fixtures
#     freeze pre-fix semantics so Phase B item 1's and item 2's before/after are
#     measurable. Do not "fix" them here.
#   --n-envs 8 --gradient-steps 4 --train-freq 1 reproduces July's
#     0.5-updates-per-transition ratio. With --n-envs 1 the same flags give 4.0,
#     an 8x difference in gradient work per sample (handoff sec 4).
set -euo pipefail

export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

MAZE=nine_rooms
BUDGET=90000            # 10,000/region; divides by 9 exactly, no truncation
PAIRS=32                # fixes the eval pair set size
EVAL_SEED=123           # <-- THE ANCHOR. must match base.yaml / submit_phaseA.sh
SEED=1                  # training seed only
DIAG_FREQ=5000
DIAG_EPS=16

mkdir -p tests/fixtures

echo "=== regions arm (${BUDGET} aggregate, $((BUDGET / 9))/region) ==="
python train.py \
  --algo sac --mode regions --maze-name "${MAZE}" \
  --horizon 200 --eval-horizon 600 --gamma 0.995 \
  --arrival-eps 0.4 --omega-max 8.0 --wall-margin 0.0 \
  --total-steps "${BUDGET}" --composition-eval-pairs "${PAIRS}" --region-eval-episodes "${PAIRS}" \
  --seed "${SEED}" --eval-seed "${EVAL_SEED}" \
  --diag-eval-freq "${DIAG_FREQ}" --diag-eval-episodes "${DIAG_EPS}" \
  --goal-reward 10 --step-pen 0.01 --collision-pen 0 \
  --switch-gate halfplane \
  --n-envs 8 --train-freq 1 --gradient-steps 32 \
  --learning-starts 2000 --buffer-size "${BUDGET}" \
  --output-dir tests/fixtures/regions

echo "=== monolith arm (matched aggregate) ==="
python train.py \
  --algo sac --mode monolith --maze-name "${MAZE}" \
  --horizon 600 --eval-horizon 600 --gamma 0.99833 \
  --arrival-eps 0.4 --omega-max 8.0 --wall-margin 0.0 \
  --total-steps "${BUDGET}" --composition-eval-pairs "${PAIRS}" --region-eval-episodes "${PAIRS}" \
  --seed "${SEED}" --eval-seed "${EVAL_SEED}" \
  --diag-eval-freq 15000 --diag-eval-episodes "${DIAG_EPS}" \
  --goal-reward 10 --step-pen 0.01 --collision-pen 0 \
  --switch-gate halfplane \
  --n-envs 8 --train-freq 1 --gradient-steps 32 \
  --learning-starts 5000 --buffer-size "${BUDGET}" \
  --output-dir tests/fixtures/monolith

echo "=== write expected.json, then verify the gate is self-consistent ==="
python -m tests.fixture_eval freeze tests/fixtures
python -m tests.fixture_eval fixtures tests/fixtures    # must be 100% immediately

cat <<'EOF'

Checks worth reading in the freeze output, in this order:

1. "mean_geodesic_dist" must be 10.572200933471322 on BOTH arms. This is the
   only value here inherited from July. If it is not, the pair set moved and
   nothing downstream is comparable -- most likely --eval-seed disagrees with
   whatever submit_phaseA.sh and the July run used. Stop and reconcile before
   committing anything.

2. "CPU reload reproduces the training run's own eval". A diff means the live
   GPU model and the reloaded CPU model disagree. A success_rate gap of exactly
   1/32 = 0.03125 is one episode flipping at the arrival boundary. Not a
   blocker, but it bounds how much tol=0 is really buying and belongs in the
   Phase A commit message.

3. The A3/D2 accounting block. Compare eval_env_steps against the two printed
   bounds. Whichever holds identifies PeriodicEvalCallback's clock
   (model.num_timesteps vs self.n_calls -- they differ by n_envs = 8), which is
   what the sec 8.3 acceptance assertion has to encode. Right now the handoff doc
   assumes num_timesteps without having run it.

Then commit everything together:

  git add -f tests/fixtures/expected.json \
             tests/fixtures/{regions,monolith}/{summary.json,resolved_config.yaml,partition.txt} \
             tests/fixtures/{regions,monolith}/models
  git commit -m "fixtures: freeze 90k eval baseline (pre-item-1 semantics)"

Ten SAC checkpoints with 256x256 nets and saved optimizer state run roughly
30-50 MB. If that is more than you want in the object store, git-lfs track
'tests/fixtures/**/*.zip' BEFORE the first add -- retrofitting LFS means a
history rewrite.
EOF