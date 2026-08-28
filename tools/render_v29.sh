#!/bin/bash
# Render mp4s + summary pngs for v29 cells, on the SAME 60 benchmark episodes the
# numbers came from. Episode k is the same initial state in every cell (the
# stratified seeds are fixed by the env digest), so arm-vs-arm videos line up.
#
#   bash tools/render_v29.sh                      # one seed of every arm, 4 eps each
#   bash tools/render_v29.sh hardmode 8 failed    # 8 failures of hardmode
#
# LOCAL DISK ONLY -- never log these to wandb.
set -e
ARM="${1:-}"; N="${2:-4}"; PICK="${3:-auto}"
SWEEP=logs/sweep_42300917

PINS="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true
  board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000
  require_settled=false push_cone_deg=30 same_room_goal_prob=1.0
  push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=6.0
  disengaged_away_deg=60"
PORT="portals=[{x:25.0,y_lo:5.0,y_hi:25.0}]"

for CELL in "${SWEEP}"/*_s0/; do
  NAME=$(basename "${CELL}")
  [ -n "${ARM}" ] && [[ "${NAME}" != *"_${ARM}_"* ]] && continue
  # bigroom trained on a 90x60 board: SB3 refuses to load it against 50x30, so it
  # must be rendered on its own board instead of the shared benchmark.
  EXTRA=""
  if [[ "${NAME}" == *bigroom* ]]; then
    EXTRA="board_w_cm=90.0 board_h_cm=60.0 same_room_goal_prob=0.0
           eval_dist_edges=[25.0,40.0,55.0] eval_episodes_per_bin=8"
    PORT="portals=[{x:45.0,y_lo:10.0,y_hi:50.0}]"
  fi
  # INTERFACE keys must come from the cell's own meta.txt: scoring a
  # contact_frame policy as finger_velocity inverted a whole result once.
  IFACE=$(grep -o 'EXTRA_OVERRIDE=.*' "${CELL}/meta.txt" | tr ' ' '\n' | grep -E \
    '^(action_interface|slip_model|slip_limit|restrict_contact_actions|mask_inactive_finger|gap_assist)=' | tr '\n' ' ')
  echo "=== ${NAME}  [${IFACE}]"
  python eval_contact.py contact=push seed=0 ${PINS} ${PORT} ${IFACE} ${EXTRA} \
    eval_ckpt="${CELL}/model.zip" \
    eval_out="logs/eval/v29_media_${NAME}.json" \
    eval_video=true eval_video_n="${N}" eval_video_pick="${PICK}" \
    eval_summary_png=true 2>&1 | grep -E 'media ->|^  all|termination:'
done
