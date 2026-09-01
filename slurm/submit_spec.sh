#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=push_spec
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-14:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v31 PUSH SPEC LADDER -- 9 cells = {spec, spec_raw, spec_settled} x seed{0,1,2}
# at 1.2M steps (~8h/cell measured). `lean` (job 42569985) is the control.
#
# THE QUESTION. Everything through v30 trained push against a POINT position
# goal with no orientation and no stopping requirement. That is not the option
# the memo's graph traverses. This sweep trains the spec version and asks what
# it costs.
#
# WHAT CHANGES, and what each change is for:
#
#   pose goals (theta_tol_deg=22.5)        Eq 13's target set is (position AND
#     orientation bin), not a point. The goal becomes (x, y, cos, sin) -- the
#     heading is a UNIT VECTOR, not an angle, so every HER relabel (which copies
#     an achieved heading) lands on the manifold and there is no +/-pi seam.
#
#   portal_goal=true                       A crossing edge's goal is a POSE DRAWN
#     FROM THE PORTAL REGION, and the object's start heading is drawn from the
#     same admissible band. MEASURED: a 10x6 object passes a 10cm gap at only
#     31.2% of orientations (band |theta| <= 28.1deg; worst-case y-extent 11.66cm
#     at 59deg). Drawing either end outside that band asks for something no skill
#     level can do -- that is edge FEASIBILITY (sec 6.4), a property of the
#     graph, not something the policy should burn budget discovering.
#     Verified: 300/300 portal goals fit the gap; with the band removed, 73/120.
#
#   guard_face=true                        The contact face is an edge parameter
#     (Eq 7) and xi shows it to the policy, so the guard must enforce it. Without
#     it a finger that walks around a corner onto ANOTHER face still satisfies
#     "is touching", i.e. the option violates the edge it was told to execute and
#     is scored a success anyway. This also explains v30's reachability surprise.
#
#   rich_obs=true                          Eq 18's observation: contact normals
#     in the OBJECT's frame (one of four values, or (0,0) for not touching),
#     per-finger force, the four nearest walls ray-cast along the object's own
#     axes, and the xi block. 37 -> 41 dims (xi grew 8 -> 12).
#
#   xi carries the SOURCE node's interface class                Eq 18's third
#     argument. The TARGET node is deliberately NOT in xi: Eq 18 splits
#     pi(a | o(s), rho(g), xi), the terminal node is what rho(g) MEANS, and HER
#     rewrites rho(g) within an episode -- a target label in xi would disagree
#     with the goal on ~80% of every relabeled batch (the v18 bug, in the one
#     block that is supposed to be immune to it).
#
#   same_room_goal_prob=0.5                Half the episodes are a pose goal
#     inside the CURRENT room, half a pose in the portal region. Same-room goals
#     are what make this genuinely goal-conditioned over object poses, and they
#     are the main source of HER signal; portal goals are the only ones the graph
#     traverses. Cross-room-only leaves nothing for a curriculum to ramp, since
#     those goals are geometrically >=15cm.
#
# ARMS (single-factor, in order):
#   spec          the above, contact_frame actions
#   spec_raw      identical but raw (vx, vy). v29 proved the contact frame
#                 load-bearing (0.217 vs 0.739) and lean_raw is at 0.000 on 3/3
#                 seeds so far. Kept because two of this sweep's changes
#                 plausibly help it: her_valid_filter gives it cleaner goals, and
#                 the affine `push` map's dead boundary does not exist in the raw
#                 space at all. VERDICT RULE: raw "works" if it clears the
#                 untrained floor on >=2 seeds. If it is still 0.000 at 1.2M with
#                 spec at a normal number, that is a RESULT about the interface,
#                 not a bug to chase.
#   spec_settled  + require_settled=true and her_valid_filter=true. Arrival now
#                 requires the object AT REST, which is what makes push
#                 composable -- a successor option cannot start from a moving
#                 object. Both flags always move together, see below.
#
# WHY her_valid_filter AND NOT her_settled. Both express "only count a goal the
# object actually came to rest at". her_settled applies it to the SCORED PAIR:
# draw a goal, then reject. her_valid_filter applies it to the CANDIDATE POOL:
# only draw from settled, guard-valid ticks. MEASURED on this exact config --
# 16.3% of ticks are settled+guard-valid, so her_settled keeps ~16% of a batch,
# while the pool filter still finds a valid tick in 74.5% of future windows.
# Same constraint, 4.6x the retained signal. It also aligns HER's implicit goal
# distribution with the option's actual target set: "places the object came to
# rest, reached without breaking the contact mode".
#
# NOT IN THIS SWEEP, deliberately:
#   Curriculum. curriculum_levels stays null. The ramp caps GOAL RADIUS, and
#   cross-room goals are geometrically >=~15cm, so levels 0-1 are unsatisfiable
#   cross-room and leak (measured: a 10.1cm cap drew a 24.6cm median). Tuning it
#   needs its own sweep on a config we trust.
#   Object-frame observations. Fingertip positions are object-RELATIVE but
#   world-ORIENTED, while the wall distances are already object-frame -- the
#   observation is internally inconsistent and rotation equivariance has to be
#   learned from data. Fixing it must move the ACTION frame too, and it strands
#   every checkpoint including lean's, so it is its own change.
#   A brake action. The finger servo already brakes at a=0. One real cost is
#   noted: under contact_frame, push=0.5*(a+1), so "stop pushing" sits at a=-1,
#   the edge of the range where a tanh-squashed Gaussian has vanishing density.
#   Relevant to spec_settled specifically; measure before adding a dimension.
#
# COMPARABILITY -- READ BEFORE SCORING.
#   The goal SPACE changed (2-D point -> 4-D pose) and the arrival test changed.
#   NO v27-v30 number is comparable to these, and the v29 floor
#   (logs/eval/v29_floor) does not apply either: it was measured on a 2-D goal.
#   REGENERATE THE FLOOR FIRST -- tools/make_untrained_ckpt.py against a
#   benchmark built from THIS goal space -- or every number here is unanchored.
#   SB3's check_for_correct_spaces bakes the goal Box into the zip, so a lean
#   checkpoint cannot even be loaded against this env. `lean` is the control by
#   NARRATIVE (what push scored without the spec), not by shared benchmark.
# ===========================================================================

set -e

ARMS=(spec spec_raw spec_settled)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# portals=[{...}] must reach Hydra through a shell variable: bash brace-expands
# [{a,b,c}] into three words, in heredocs and sbatch scripts alike.
PORT="portals=[{x:25.0,y_lo:10.0,y_hi:20.0}]"

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
IFACE="action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 mask_inactive_finger=false gap_assist=false"
TASK="push_cone_deg=90 disengaged_away_deg=60 push_range_min_cm=3.0 object_theta_spread_deg=90 angular_drag_arm_cm=3.12 board_w_cm=50.0 board_h_cm=30.0 same_room_goal_prob=0.5"
SPEC="theta_tol_deg=22.5 theta_goal_window_deg=45.0 portal_goal=true portal_clearance_cm=0.5 guard_face=true rich_obs=true"
SETTLE="require_settled=false her_settled=false her_valid_filter=false"

case "${ARM}" in
  spec)     ;;                                   # base, unmodified
  # gap_assist and slip_* are contact_frame-only keys, so the raw arm drops
  # them rather than setting them false: finger_velocity has never had the gap
  # assist, which is why `false` is the midpoint of full -> raw and not raw.
  spec_raw) IFACE="action_interface=finger_velocity mask_inactive_finger=false" ;;
  # The two settle flags MOVE TOGETHER on purpose. require_settled alone makes
  # arrival strictly harder while HER keeps relabeling to moving ticks, so the
  # buffer teaches a target the reward will not pay for. eps_v/eps_omega must be
  # set for either to mean anything -- object_settled falls back to its module
  # defaults when they are None, which would be a silent third change.
  spec_settled) SETTLE="require_settled=true her_settled=false her_valid_filter=true eps_v_cm_s=0.5 eps_omega_deg_s=5.0" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="push"
TOTAL_STEPS=1200000
CKPT_FREQ=400000
EXTRA_OVERRIDE="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true ${PORT} min_progress_ticks=1 her_n_sampled_goal=4 learning_starts=10000 target_clip=10 ckpt_freq=${CKPT_FREQ} ${TASK} ${SPEC} ${SETTLE} ${IFACE}"
RUN_TAG="push_${ARM}_s${SEED}"
GROUP="spec"

source slurm/_run_cell.sh
