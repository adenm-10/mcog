#!/bin/bash
# BAR 2, ZERO-SHOT: does push already SETTLE at the goal, with no training?
#
# Bar 2 (docs/TODO.md) is Eq 13's settled arrival -- the object at the goal AND
# at rest -- and it is what makes a push composable at all: the next option's
# initiation set is a state, not a fly-through. Every v32/v33 push cell trained
# and scored with require_settled=false.
#
# WHY A FROZEN CHECKPOINT CAN ANSWER THIS. require_settled changes the ARRIVAL
# TEST and nothing the policy reads: it is absent from obs(), from xi and from
# the action space, and it does not touch the reset sampler. So the stratified
# seeds draw the SAME 60 initial states and the comparison is per-episode. It
# does change what success IS, so it is a TASK key, the digest moves, and this
# is a DIFFERENT BENCHMARK that needs its own floor -- which is cell 0 below.
# (CLAUDE.md: "regenerate the untrained floor for every protocol change".)
#
# The cheap check exists because ctl's median final distance is already 0.39cm
# against a 0.40cm tolerance (v33 scoring), so Bar 2 may need no training at
# all -- and Bar 2 is the only thing blocking Phase B. Minutes, not GPU-hours.
#
# Reads out as: settled success vs the 0.674 position-only number, on the same
# episodes. The gap IS the price of settling.
set -e

OUT=logs/eval/v34_bar2
mkdir -p "$OUT"
SWEEP=logs/sweep_43679344          # v33
PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# v33's OWN benchmark protocol, copied from its PINS.txt with require_settled
# flipped and nothing else touched -- so the only thing that can move a number
# is the settle requirement. curriculum_levels=null is the full range, which is
# what every arm's last level trained on.
PROTO="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 \
her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 \
push_cone_deg=30 same_room_goal_prob=0.5 \
push_range_min_cm=null object_theta_spread_deg=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
guard_face=false rich_obs=true push_range_max_cm=null \
curriculum_mode=band curriculum_levels=null \
theta_tol_deg=22.5 theta_goal_window_deg=45.0"

# INTERFACE keys, read from the cells' own meta.txt (all three ctl cells share
# them), and OUTSIDE the digest -- so the same protocol scores v1 checkpoints.
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=true gap_assist=false obs_version=1 \
normalize_goal_keys=false"

score () {   # label, ckpt, require_settled
  local NAME="$1" CKPT="$2" RS="$3"
  local DIR="$OUT/$NAME"
  mkdir -p "$DIR"
  echo "=== $NAME  (require_settled=$RS) ==="
  python eval_contact.py contact=push seed=0 \
    $PROTO $IFACE "$PORT" require_settled=$RS \
    eval_ckpt="$CKPT" eval_out="$DIR/eval.json"
}

# Cell 0: THE FLOOR for this protocol. Not optional -- a settled-arrival floor
# is not the position-arrival floor, and nothing on disk measures it.
mkdir -p "$OUT/floor_settled"
python tools/make_untrained_ckpt.py contact=push seed=0 \
  $PROTO $IFACE "$PORT" require_settled=true \
  eval_out="$OUT/floor_settled/model.zip"
score floor_settled "$OUT/floor_settled/model.zip" true

# Cells 1-6: the three v33 ctl seeds, BOTH checkpoints, settled.
# Both, because a claim resting on one unconverged checkpoint is not a claim
# (CLAUDE.md); v33's argmax sat at 555k-595k of 600k for every arm.
for s in 0 1 2; do
  CELL=$(ls -d ${SWEEP}/*_push_ctl_s${s} 2>/dev/null | head -1)
  [ -n "$CELL" ] || { echo "no ctl_s${s} in ${SWEEP}" >&2; exit 2; }
  score "ctl_s${s}_settled"      "${CELL}/model.zip"      true
  score "ctl_s${s}_best_settled" "${CELL}/model_best.zip" true
done

# Cells 7-8: the SAME checkpoint on the position-only protocol, re-scored HERE
# rather than quoted from v33. The comparison has to come off one code version:
# quoting 0.683 from an archived report and 0.5xx from this run would attribute
# every change since v33 to the settle requirement.
for s in 0 1 2; do
  CELL=$(ls -d ${SWEEP}/*_push_ctl_s${s} 2>/dev/null | head -1)
  score "ctl_s${s}_position" "${CELL}/model.zip" false
done

echo
echo "=== READ IT LIKE THIS ==="
echo "ctl_s*_position  is the >=3cm mean on v33's own protocol (digest 249434216cd2)."
echo "ctl_s*_settled   is the same episodes under Eq 13's settled arrival."
echo "The DIFFERENCE is Bar 2's price. floor_settled bounds what is free."
echo "Both digests are printed above; they MUST differ (require_settled is a"
echo "task key) and each settled cell must match floor_settled's digest."

cat > "$OUT/PROTOCOL.md" <<EOF
# Bar 2 zero-shot protocol (v34)

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/bar2_zeroshot.sh\`.
No training: v33's frozen \`ctl\` checkpoints, re-scored.

TASK pins (inside the env digest), from \`logs/sweep_43679344/PINS.txt\` with
\`require_settled\` flipped and nothing else touched:

\`\`\`
$PROTO
$PORT
\`\`\`

INTERFACE block (outside the digest, read from the cells' own \`meta.txt\`):

\`\`\`
$IFACE
\`\`\`

## Digests

| \`require_settled\` | digest |
|---|---|
| \`false\` | \`249434216cd2\` -- v33's own, so \`*_position\` is directly comparable to every archived v32/v33 number |
| \`true\`  | \`fdc2a41dc665\` -- a DIFFERENT benchmark; \`require_settled\` is a task key |

## Why a frozen checkpoint answers this

\`require_settled\` changes the ARRIVAL TEST and nothing the policy reads: it is
absent from \`obs()\`, from \`xi\` and from the action space, and it does not touch
the reset sampler. The stratified seeds therefore draw the SAME 60 initial
states under both digests, so the comparison is per-episode rather than
distributional. Trajectories diverge only after the first position arrival,
which is exactly the thing being measured.

## Primary metric

Mean success on goals >=3cm, the same restriction v33 reported, under BOTH
\`model.zip\` and \`model_best.zip\`.
EOF
echo "PROTOCOL.md written to $OUT"
