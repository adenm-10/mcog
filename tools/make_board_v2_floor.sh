#!/bin/bash
# Untrained floors for BOARD V2 -- Sweep C's four interface arms and Sweep D's
# two Gamma arms.
#
# A floor is specific to (interface, goal space, protocol) and NEVER transfers
# (CLAUDE.md), so this regenerates every one from scratch. Run BEFORE the
# sweeps: a floor produced after the numbers is a floor chosen to fit them.
#
# THE PROTOCOL IS SOURCED, NOT RETYPED. make_v32_floor.sh omitted
# `disengaged_away_deg` once and produced a floor on a different reset
# distribution than the sweep it bounded (1a72f6438f34 vs 249434216cd2). There
# is exactly one copy of board v2's pins and it lives in slurm/pins_board_v2.sh.
#
# BOARD V2 NEEDS THREE FLOORS PER CELL, NOT ONE. The benchmark reports two
# predicates on the same episodes (see slurm/pins_board_v2.sh), and their floors
# are an order of magnitude apart:
#   same-room, pose reached      -- a random policy has to land a pose. Low.
#   crossing,  far-room pose     -- likewise, and further away. Low.
#   crossing,  entered dst room  -- a random policy SHOVES the object through an
#                                   open doorway. Measured 0.271 overall / 0.42
#                                   in the crossing bins on board v1. HIGH, and
#                                   pooling it with the pose numbers would inflate
#                                   the headline for free.
# tools/split_floor.py prints all three from one eval.
set -e

cd "$(dirname "$0")/.."
source slurm/pins_board_v2.sh
source slurm/pins_gamma_ladder.sh

OUT=logs/eval/board_v2_floor
mkdir -p "$OUT"

CF="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=true gap_assist=false"
RAW="action_interface=finger_velocity slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=true gap_assist=false"

run_cell () {   # name, PINS, template, extra overrides...
  local NAME="$1" PINS="$2" TMPL="$3"; shift 3
  local DIR="$OUT/$NAME"
  mkdir -p "$DIR"
  echo "=== $NAME ($TMPL) ==="
  python tools/make_untrained_ckpt.py "contact=$TMPL" seed=0 \
    $PINS "$@" eval_out="$DIR/model.zip"
  python eval_contact.py "contact=$TMPL" seed=0 \
    $PINS "$@" \
    eval_ckpt="$DIR/model.zip" eval_out="$DIR/eval.json"
}

# --- Sweep C: four interface arms, ONE task, so ONE digest across all four ----
# Each still needs its OWN floor: an untrained net reads a different vector (and
# `raw` emits different actions), so its random policy is a different random
# policy even on an identical task.
V2="obs_version=2 rich_obs=true normalize_goal_keys=false"
V1="obs_version=1 rich_obs=true normalize_goal_keys=false"

run_cell ctl    "$BOARD_V2_PINS" push $V2 $CF  xi_gamma_mode=face
run_cell count  "$BOARD_V2_PINS" push $V2 $CF xi_gamma_mode=count \
                guard_contact_count=1 mask_inactive_finger=false
run_cell raw    "$BOARD_V2_PINS" push $V2 $RAW xi_gamma_mode=face
run_cell obs_v1 "$BOARD_V2_PINS" push $V1 $CF  xi_gamma_mode=face

# --- Sweep D: the Gamma ladder. Recontact has no board, so board_w/h and the
# portals are inert here; what matters is v_max, the horizon and the guard.
# Each arm moves a TASK key (which Gamma classes are drawn), so each gets its own.
# Sourced from slurm/pins_gamma_ladder.sh -- recontact has no board, and
# ContactEnv REFUSES board v2's push-only keys rather than ignoring them.
# finger_velocity, not contact_frame: contact_frame is push-only (it maintains a
# contact that recontact, by definition, does not yet have) and ContactEnv
# refuses it rather than ignoring it. mask_inactive_finger=false because a
# two-contact Gamma needs BOTH fingers to move.
GAMMA_IFACE="obs_version=2 normalize_goal_keys=false xi_gamma_mode=count \
action_interface=finger_velocity slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=false gap_assist=false"

run_cell g_one      "$GAMMA_PINS" recontact $GAMMA_IFACE \
                    "$GAMMA_GOAL_ONE" "$GAMMA_INIT_ONE"
run_cell g_two_disp "$GAMMA_PINS" recontact $GAMMA_IFACE \
                    "$GAMMA_GOAL_TWO" "$GAMMA_INIT_TWO"

echo
echo "=== FLOORS, split by stratum and predicate ==="
python tools/split_floor.py "$OUT"

cat > "$OUT/PROTOCOL.md" <<EOF
# Board v2 floor protocol

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/make_board_v2_floor.sh\`,
zero gradient steps. Task pins sourced from \`slurm/pins_board_v2.sh\`, never
retyped:

\`\`\`
$BOARD_V2_PINS
\`\`\`

Six cells. The four push arms share ONE digest (their differences are all
INTERFACE keys) and still carry four separate floors, because an untrained net
reads a different vector per interface. The two Gamma arms each move a task key
and so carry their own digest as well as their own floor.

Read the crossing/entered-dst floor separately from the pose floors. They are
not the same number and must not be pooled.
EOF
echo "wrote $OUT/PROTOCOL.md"
