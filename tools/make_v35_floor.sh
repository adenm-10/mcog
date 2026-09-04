#!/bin/bash
# Untrained floors for SWEEP B's four protocols.
#
# A floor is specific to (interface, goal space, protocol) and NEVER transfers
# (CLAUDE.md), so a sweep that moves an interface key OR a task key needs its
# own. Sweep B moves one of each:
#
#   ctl       obs v1, centre protocol, cone 30, no spread.
#             THE CONTROL, and a regression check: it must reproduce
#             logs/eval/v34_floor/a1_v1_centre's 0.042 on goals >=3cm at digest
#             249434216cd2. If it does not, something moved between Sweep A and
#             here and no Sweep B number can be attributed.
#   obsv2     obs v2 + normalized goal keys, same protocol. Same DIGEST as ctl
#             (both keys are INTERFACE keys) but NOT the same floor: an untrained
#             net reads a different vector, so its random policy is a different
#             random policy. logs/eval/v34_floor/a3_v2_along is obs v2 on the
#             ALONG protocol and does not transfer here.
#   widecone  cone 90, own protocol. push_cone_deg is a TASK key, so this is a
#             different goal distribution and a different digest.
#   spread    spread 90, own protocol. Likewise, and it additionally rotates the
#             face offset, the face normal and the coned goal direction.
#
# Zero gradient steps, ~35s per eval. Run BEFORE the sweep: a floor produced
# after the numbers is a floor chosen to fit them.
set -e

OUT=logs/eval/v35_floor
mkdir -p "$OUT"

PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# THE PROTOCOL. Derived from Sweep B's own PINS -- slurm/submit_sweep.sh writes
# the identical line to <sweep>/PINS.txt -- rather than retyped. The first
# version of make_v32_floor.sh omitted disengaged_away_deg, a TASK key inside
# the digest, and produced a floor on a different reset distribution than the
# sweep (1a72f6438f34 vs 249434216cd2). VERIFY THE DIGEST, ALWAYS.
PROTO="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
board_w_cm=50.0 board_h_cm=30.0 min_progress_ticks=1 learning_starts=10000 \
her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 \
require_settled=false same_room_goal_prob=0.5 \
push_range_min_cm=null angular_drag_arm_cm=3.12 \
portal_arrival=false portal_goal=true portal_clearance_cm=0.5 \
guard_face=false push_range_max_cm=null \
curriculum_mode=band curriculum_levels=null \
theta_tol_deg=22.5 theta_goal_window_deg=45.0 push_spawn_along_frac=null"

CF="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=true gap_assist=false"

run_cell () {   # name, obs-block, task-block
  local NAME="$1" OBS="$2" TASK="$3"
  local DIR="$OUT/$NAME"
  mkdir -p "$DIR"
  echo "=== $NAME ==="
  python tools/make_untrained_ckpt.py contact=push seed=0 \
    $PROTO $OBS $TASK $CF "$PORT" eval_out="$DIR/model.zip"
  python eval_contact.py contact=push seed=0 \
    $PROTO $OBS $TASK $CF "$PORT" \
    eval_ckpt="$DIR/model.zip" eval_out="$DIR/eval.json"
}

V1="obs_version=1 rich_obs=true normalize_goal_keys=false"
V2="obs_version=2 rich_obs=true normalize_goal_keys=true"
BASE="push_cone_deg=30 object_theta_spread_deg=null"
WIDE="push_cone_deg=90 object_theta_spread_deg=null"
SPRD="push_cone_deg=30 object_theta_spread_deg=90"

run_cell ctl      "$V1" "$BASE"
run_cell obsv2    "$V2" "$BASE"
run_cell widecone "$V1" "$WIDE"
run_cell spread   "$V1" "$SPRD"

cat > "$OUT/PROTOCOL.md" <<EOF
# Sweep B floor protocol (v35)

Generated $(date -u +%Y-%m-%dT%H:%M:%SZ) by \`tools/make_v35_floor.sh\`,
zero gradient steps.

Common TASK pins (inside the env digest), identical to the line
\`slurm/submit_sweep.sh\` writes to \`<sweep>/PINS.txt\`:

\`\`\`
$PROTO
$PORT
\`\`\`

Per cell, exactly one block differs from \`ctl\`:

| cell | differs by | class |
|---|---|---|
| \`ctl\` | nothing | -- |
| \`obsv2\` | \`obs_version=2 normalize_goal_keys=true\` | INTERFACE (digest unchanged) |
| \`widecone\` | \`push_cone_deg=90\` | TASK (own digest) |
| \`spread\` | \`object_theta_spread_deg=90\` | TASK (own digest) |

## Measured floors

| cell | digest | >=3cm | all 5 bins |
|---|---|---|---|
$(for n in ctl obsv2 widecone spread; do
python - "$OUT/$n/eval.json" "$n" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
eps = [e for e in d["episodes"] if e["d0"] >= 3.0]
s3 = sum(e["success"] for e in eps) / len(eps)
print(f"| {sys.argv[2]} | {d['env_digest']} | {s3:.3f} | {d['success']:.3f} |")
PY
done)

\`ctl\` must read 0.042 on goals >=3cm at digest \`249434216cd2\`, matching
\`logs/eval/v34_floor/a1_v1_centre\`. \`obsv2\` shares that digest by design and
is listed separately because the floor, unlike the digest, depends on the
observation the untrained net reads.

## Why the primary metric is restricted to >=3cm

20% of the stratified set is under 3cm, where success needs less than 1cm of
object motion, so the 5-bin mean carries a 0.150 floor from that bin alone.
Reported here for both, used for one.
EOF
echo "PROTOCOL.md written to $OUT"
