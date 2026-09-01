#!/bin/bash
# Untrained floor for the v32 curriculum sweep's goal spaces.
#
# A floor is specific to (interface, goal space, protocol), and v32 changes all
# three against logs/eval/v29_floor: srg 1.0 -> 0.5, damping 6.00 -> 3.12,
# portal 20cm -> 10cm, and arrival becomes the PORTAL SET on crossing edges
# (portal_arrival, which no earlier sweep ever enabled). So no earlier floor
# transfers and every v32 number is unanchored until this runs.
#
# 4 cells = {2-D position goal, 4-D pose goal} x {contact_frame, finger_velocity}.
# ~34s per eval. Zero gradient steps.
#
# The benchmark uses the REVERSE sampler at its full range
# (curriculum_mode=band curriculum_levels=null), which is what every arm's last
# level trains on. Scoring on the forward sampler instead would leave a small
# train/test distribution shift for no reason -- both cost the same 3 minutes.
set -e

OUT=logs/eval/v32_floor
mkdir -p "$OUT"

PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# THE v32 PROTOCOL. Pinned here, and copied into $OUT/PROTOCOL.md next to the
# numbers, because a protocol living only in a launcher comment is what made
# every v25 cross-version comparison wrong.
# DERIVED FROM A LAUNCHED CELL, not retyped: this is
# logs/sweep_43572361/*push_base_s0*/meta.txt's EXTRA_OVERRIDE minus the
# per-arm INTERFACE and curriculum keys. The first version of this file omitted
# disengaged_away_deg, which is a TASK key inside the env digest, and the floor
# came out on a different reset distribution than the sweep (digest
# 1a72f6438f34 vs the sweep's 249434216cd2). Verify the digest, always.
PROTO="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 \
her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 \
require_settled=false push_cone_deg=30 same_room_goal_prob=0.5 \
push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
guard_face=false rich_obs=true push_range_max_cm=null \
curriculum_mode=band curriculum_levels=null"

# Goal space is set by theta_tol_deg: null -> 2-D position, 22.5 -> 4-D pose.
# Eq 11's eight orientation bins make 45deg bins, i.e. +/-22.5deg.
for GOAL in pos pose; do
  case "$GOAL" in
    pos)  GSPEC="theta_tol_deg=null" ;;
    pose) GSPEC="theta_tol_deg=22.5 theta_goal_window_deg=45.0" ;;
  esac
  for IFACE_NAME in contact_frame finger_velocity; do
    case "$IFACE_NAME" in
      contact_frame)   IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=true gap_assist=false" ;;
      finger_velocity) IFACE="action_interface=finger_velocity mask_inactive_finger=true" ;;
    esac
    TAG="${GOAL}_${IFACE_NAME}"
    echo "=== $TAG ==="
    python tools/make_untrained_ckpt.py contact=push seed=0 \
      $PROTO $GSPEC $IFACE "$PORT" eval_out="$OUT/untrained_${TAG}.zip"
    python eval_contact.py contact=push seed=0 \
      $PROTO $GSPEC $IFACE "$PORT" \
      eval_ckpt="$OUT/untrained_${TAG}.zip" eval_out="$OUT/untrained_${TAG}.json"
  done
done

cat > "$OUT/PROTOCOL.md" <<EOF
# v32 floor protocol

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by tools/make_v32_floor.sh, zero gradient steps.

TASK pins (inside the env digest):

\`\`\`
$PROTO
$PORT
\`\`\`

Per-cell keys: theta_tol_deg (null = 2-D position goal, 22.5 = 4-D pose goal)
and the INTERFACE block (taken from each cell's meta.txt when scoring a real sweep).

## Why this is not logs/eval/v29_floor

| key | v29 floor | v32 |
|---|---|---|
| same_room_goal_prob | 1.0 | **0.5** -- v32 trains both goal types, so the benchmark must score both |
| angular_drag_arm_cm | 6.00 | **3.12** -- v29 physdamp; 6.00 is above the physical ceiling |
| portal | y 5-25 (20cm) | **y 10-20 (10cm)** -- matches training |
| portal_arrival | false | **true** -- crossing edges arrive at the PORTAL SET (Eq 13), not a 0.4cm point |
| portal_goal | false | **true** |
| rich_obs | false | **true** |

Report success on goals >=3cm beside the 5-bin mean: 20% of the set sits under
3cm where the floor itself scores ~0.75, which halves every arm difference.
EOF
echo
echo "wrote $OUT/PROTOCOL.md"
