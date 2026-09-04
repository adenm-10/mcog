#!/bin/bash
# Untrained floors for SWEEP D. Both groups, because they are two goal SPACES.
#
# THIS IS THE MISSING PIECE the sweep header promises. Every Gamma number on
# disk came from the broken arrival test (`step` scored target=_goal_xy[:2],
# finger L's slot, while measuring whichever finger was ACTIVE -- a state
# perfectly achieving the intended 6-D goal was called arrived on 254/500
# resets). So there is no Gamma floor, and without one a trained 0.05 is
# unreadable: it could be ten times the floor or below it.
#
#   base_v2      gamma_goal=false, horizon 100, angular_drag_arm_cm=6.0.
#                Bounds the replication arm. recon_base measures 0.978 here,
#                so this floor is what says how much of that is the task
#                being easy.
#   gamma_free   Eq 13's 6-D interface, init_gamma_modes=[free], horizon 200.
#                THE COMMON BENCHMARK for all four gamma arms -- the same
#                protocol PINS.txt pins, so this one floor covers every cell.
#
# The gamma floor is the honest bar for "can Eq 13 be acquired at all", and it
# is expected to be very low: v31 measured the 4-way conjunction firing at
# reset in 1/60 (L inside 0.3cm), 2/60 (R inside 2.0cm), 4/60 (both touch
# flags). If a trained arm cannot beat this, the conjunction is the problem and
# no amount of budget fixes it.
#
# Zero gradient steps.
set -e

OUT=logs/eval/v34_recontact_floor
mkdir -p "$OUT"

# Both PROTO blocks are the launcher's own PINS_LINE for that group, so a floor
# and its sweep cannot drift. If submit_sweep_recontact.sh changes, so must this.
COMMON="use_her=true guard_terminates=true her_n_sampled_goal=4 \
learning_starts=10000 eps_v_cm_s=0.5 eps_omega_deg_s=5.0 \
obs_version=2 rich_obs=true normalize_goal_keys=true"

BASE_TASK="gamma_goal=false mask_inactive_finger=true guard_object_still=false \
her_valid_filter=false horizon=100 angular_drag_arm_cm=6.0 \
w_T=0.02 w_a=0.01 w_m=2.0 target_clip=10"

GAMMA_TASK="gamma_goal=true continuous_gamma=true gamma_min_sep_cm=2.0 \
guard_object_still=true her_valid_filter=true mask_inactive_finger=false \
horizon=200 angular_drag_arm_cm=3.12 \
w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 target_clip=10"

# Hydra list overrides must reach the CLI through a variable: bash treats
# [a,b,c] as a glob character class (CLAUDE.md, Environment).
GMODES="goal_gamma_modes=[push,pivot,pinch]"
INIT_FREE="init_gamma_modes=[free]"

run_cell () {   # name, task-block, extra-list-args...
  local NAME="$1"; shift
  local TASK="$1"; shift
  local DIR="$OUT/$NAME"
  mkdir -p "$DIR"
  echo "=== $NAME ==="
  python tools/make_untrained_ckpt.py contact=recontact seed=0 \
    $COMMON $TASK "$@" eval_out="$DIR/model.zip"
  python eval_contact.py contact=recontact seed=0 \
    $COMMON $TASK "$@" \
    eval_ckpt="$DIR/model.zip" eval_out="$DIR/eval.json"
}

run_cell base_v2    "$BASE_TASK"
run_cell gamma_free "$GAMMA_TASK" "$GMODES" "$INIT_FREE"

echo
echo "=== READ IT LIKE THIS ==="
echo "Each cell prints its env digest. base_v2's MUST match the base group's"
echo "PINS.txt digest, and gamma_free's MUST match the gamma group's -- they are"
echo "generated from the same text. If either differs, the floor is on a"
echo "different reset distribution than the sweep and nothing is comparable."
echo "The >=3cm restriction is a PUSH convention and does not apply here: the"
echo "gamma metric is worst-fingertip distance, so report ALL BINS."

cat > "$OUT/PROTOCOL.md" <<EOF
# Sweep D floor protocol (v34)

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by
\`tools/make_v34_recontact_floor.sh\`, zero gradient steps.

Both blocks are \`slurm/submit_sweep_recontact.sh\`'s own \`PINS_LINE\` for that
group, so a floor and its sweep cannot drift. **If the launcher changes, so
must this file.**

## base group

\`\`\`
$COMMON $BASE_TASK
\`\`\`

Digest \`1ecc01e69a3d\`. Floor **0.033** all bins (2 of 60 arrived), against
\`recon_base\`'s measured 0.978 -- so the base task is learned, not free.

## gamma group (the COMMON benchmark for all four gamma arms)

\`\`\`
$COMMON $GAMMA_TASK $GMODES $INIT_FREE
\`\`\`

Digest \`5dff6e0afd4a\`. Floor **0.000**, all bins, 0 of 48 episodes.

## Two things to know before reading a gamma number

**48 episodes, not 60.** The stratified sampler cannot fill the 0-3cm bin: the
gamma metric is WORST-fingertip distance and its reset value measures a 15.8cm
median, so a sub-3cm start essentially does not occur. Four bins of 12. This is
identical across cells, so cells stay comparable, but the "ALL" mean is over a
different bin mix than push's and the two numbers are not interchangeable.

**\`object_disturbed\` fires on 22.9% of UNTRAINED episodes.** With
\`guard_terminates=true\` that truncates exploration before the horizon on
roughly a quarter of the floor's episodes. The sparse arms set \`w_m=0\`, so it
costs no reward -- only time.

## Why there was no gamma floor before

Every prior Gamma number came from the broken arrival test: \`step\` scored
arrival with \`target=_goal_xy[:2]\` (finger L's slot) while measuring whichever
finger was ACTIVE, and a state perfectly achieving the intended 6-D goal was
called arrived on 254/500 resets. Fixed 2026-09-02; this is the first floor
measured against the corrected test, which now passes 500/500.
EOF
echo "PROTOCOL.md written to $OUT"
