#!/bin/bash
# Untrained floor for SWEEP A's protocol -- the randomized along-face spawn.
#
# A floor is specific to (interface, goal space, protocol) and NEVER transfers
# (CLAUDE.md). Sweep A changes the protocol in exactly one TASK way against
# logs/eval/v32_floor: push_spawn_along_frac null -> 0.7, which moves the reset
# distribution and the env digest (249434216cd2 -> 646ba4ae1fd4). Everything
# else about the task is byte-identical, deliberately, so the spawn change is
# the only thing the A2-A1 comparison can be attributing.
#
# 4 cells, one directory each so its vecnormalize.pkl is unambiguous:
#
#   a1_v1_centre     obs v1, raw goal keys, face-CENTRE spawn.
#                    The CONTROL. Must reproduce logs/eval/v32_floor's
#                    0.042 on goals >=3cm -- if it does not, something other
#                    than the spawn moved and Sweep A is not readable.
#   a2_v1_along      obs v1, raw goal keys, along-face spawn.
#   a3_v2_along      obs v2 + normalized goal keys, along-face spawn.
#                    Sweep A's A3 arm, i.e. the protocol B/C/D all inherit.
#   a3_v2_along_raw  as a3, but action_interface=finger_velocity.
#                    The RAW-ACTION floor, which Sweep C's ppo_raw arm needs:
#                    v29 measured 0.000 on goals >=3cm for raw actions against
#                    0.150 all-bins, and that gap is the whole reason >=3cm is
#                    the primary metric.
#
# Zero gradient steps. ~35s per eval.
set -e

OUT=logs/eval/v34_floor
mkdir -p "$OUT"

PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# THE PROTOCOL. Derived from logs/sweep_43679344/PINS.txt (v33's own benchmark)
# rather than retyped, plus the one new TASK key. The first version of
# make_v32_floor.sh omitted disengaged_away_deg -- a TASK key inside the digest
# -- and produced a floor on a different reset distribution than the sweep
# (1a72f6438f34 vs 249434216cd2). VERIFY THE DIGEST, ALWAYS.
PROTO="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 \
her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 \
require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 \
push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
guard_face=false push_range_max_cm=null \
curriculum_mode=band curriculum_levels=null \
theta_tol_deg=22.5 theta_goal_window_deg=45.0"

CF="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=true gap_assist=false"
FV="action_interface=finger_velocity mask_inactive_finger=true gap_assist=false"

run_cell () {   # name, obs-block, spawn-block, iface-block
  local NAME="$1" OBS="$2" SPAWN="$3" IFACE="$4"
  local DIR="$OUT/$NAME"
  mkdir -p "$DIR"
  echo "=== $NAME ==="
  python tools/make_untrained_ckpt.py contact=push seed=0 \
    $PROTO $OBS $SPAWN $IFACE "$PORT" eval_out="$DIR/model.zip"
  python eval_contact.py contact=push seed=0 \
    $PROTO $OBS $SPAWN $IFACE "$PORT" \
    eval_ckpt="$DIR/model.zip" eval_out="$DIR/eval.json"
}

V1="obs_version=1 rich_obs=true normalize_goal_keys=false"
V2="obs_version=2 rich_obs=true normalize_goal_keys=true"
CENTRE="push_spawn_along_frac=null"
ALONG="push_spawn_along_frac=0.7"

run_cell a1_v1_centre    "$V1" "$CENTRE" "$CF"
run_cell a2_v1_along     "$V1" "$ALONG"  "$CF"
run_cell a3_v2_along     "$V2" "$ALONG"  "$CF"
run_cell a3_v2_along_raw "$V2" "$ALONG"  "$FV"

cat > "$OUT/PROTOCOL.md" <<EOF
# Sweep A floor protocol (v34)

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/make_v34_floor.sh\`,
zero gradient steps.

TASK pins (inside the env digest):

\`\`\`
$PROTO
$PORT
\`\`\`

Per-cell: \`push_spawn_along_frac\` (null = face centre, 0.7 = randomized
along-face) and the INTERFACE block, which sits OUTSIDE the digest and is read
per cell from each run's own \`meta.txt\` when scoring a real sweep.

## Expected digests

| spawn | digest |
|---|---|
| \`push_spawn_along_frac=null\` | \`249434216cd2\` (identical to v32/v33 -- a defaulted post-hoc TASK key is omitted from the stamp, so the archived anchor survives) |
| \`push_spawn_along_frac=0.7\` | \`646ba4ae1fd4\` |

\`obs_version\` and \`normalize_goal_keys\` are INTERFACE keys and do NOT move
the digest, so A1/A2/A3 are scorable against one another once the spawn matches.

## Why this is not logs/eval/v32_floor

v32's floor was measured at the face-CENTRE spawn. The centre is not a neutral
choice: a contact there pushing along the inward normal produces EXACTLY ZERO
torque, because the lever arm is parallel to the force. Randomizing it changes
both what a random policy achieves and what a trained one CAN achieve, so the
old floor does not transfer.

## Reading it

\`a1_v1_centre\` is the control and must reproduce v32's 0.042 on goals >=3cm.
If it does not, something other than the spawn moved between v33 and Sweep A,
and no A2-A1 or A3-A2 difference can be attributed until that is found.
EOF
echo "PROTOCOL.md written to $OUT"
