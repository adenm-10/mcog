#!/bin/bash
# slurm/pins_board_v2.sh -- THE one definition of board v2's task protocol.
#
# Sourced by the launcher, the floor builder and the rung scorer. Retyping a
# protocol has cost this project a floor once already: make_v32_floor.sh omitted
# `disengaged_away_deg` and produced digest 1a72f6438f34 against the sweep's
# 249434216cd2, so the "untrained floor" bounded a different task than the one
# the policies were scored on. There is no second copy of this string.
#
# TASK keys only. Interface keys (action_interface, obs_version, xi_gamma_mode,
# ...) are per ARM and are read from each run's own metadata -- putting one here
# would silently remove it from the env digest.
#
# ---------------------------------------------------------------------------
# WHAT BOARD V2 IS, AND WHY (decision D4, plus the 2026-09-08 pose/arrival call)
#
#   90x60, 3 rooms, doors 13.0cm offset 34cm in y.
#     The gap is DERIVED, not swept: the object's diagonal is
#     sqrt(10^2 + 6^2) = 11.66cm, so gap >= diagonal + clearance admits EVERY
#     orientation and the door becomes a routing constraint instead of an
#     orientation lottery. Measured against board v1: cross-room goals within
#     3cm of the wall plane 1.000 -> 0.000, crossing distance 11.2 -> 28.7cm,
#     straight path blocked by a wall 0.000 -> 0.352, regions/edge types 2/4 -> 3/7.
#
#   portal_goal=false.
#     v1 put the goal IN the doorway, so "cross the room" only ever asked the
#     object to reach the wall plane. False makes the goal land in the far room.
#     THIS IS WHAT EXPOSED THE SAMPLER BUG: _sample_push_edge_reverse never
#     checked the object->goal ray passes the doorway, and with portal_goal=false
#     73.5% of crossing resets had no straight-line path. Fixed 2026-09-08;
#     re-measured 0 of 199 crossing resets blocked.
#
#   portal_arrival=false AT TRAINING, and the crossing predicate applied at
#   EVAL only.
#     "Leave through the correct portal" is the faithful job of a crossing edge
#     inside a route. But `crossed()` is a property of the TRAJECTORY, not of
#     (achieved, desired), while HER grades ~80% of every batch by
#     pose_arrived on a relabeled goal. Training on it would put the rollout
#     reward and the relabeled reward on two different objectives -- the exact
#     divergence that made 63 GPU-hours of Gamma arms uninterpretable. So the
#     policy trains on a well-posed pose goal in the far room, and the eval
#     reports BOTH predicates. See tools/score_board_v2.sh.
#
#   object_theta_spread_deg=180 -- full START-pose randomization.
#     The GOAL heading window stays at 45: push produces a median 1.8deg of
#     object rotation per episode (p90 7.0), so a uniform goal heading would be
#     unreachable in ~84% of episodes. Arbitrary reorientation is `pivot`'s job
#     (Eq 4), and pivot is not implemented. Sweep B priced spread=90 at about
#     -0.06 with a CI touching zero; 180 is expected to cost more and that price
#     is the point, not a surprise.
#
#   push_cone_deg=75 -- SET FROM MEASUREMENT, never swept.
#     tools/probe_reachable.py, 1024 open-loop rollouts: the p90 of |direction
#     off the inward normal| is 74.2deg under guard_face=adjacent. +/-180 is NOT
#     supportable -- only 1.4% of reachable directions land in the 120-180 bin
#     with the guard on, and most apparent "behind the contact" reachability was
#     bought by walking to another face.
#
#   require_settled=true -- Bar 2's condition. A successor option cannot start
#     from a moving object. Met zero-shot at 0.625 for a measured price of 0.049.
#
#   same_room_goal_prob=0.5 -- half the episodes are goal-conditioned inside the
#     current region, which on the bigger board now reaches 41cm (v1 max: 14.4).
#
#   guard_face=false -- D5. `adjacent` costs 0.042-0.083 and IS enforceable, but
#     an interface comparison between two floored arms answers nothing, so it is
#     re-scored zero-shot on every cell instead of trained.
# ---------------------------------------------------------------------------

# Hydra list-of-dict overrides MUST reach the CLI through a shell variable:
# bash brace-expands [{a,b}] into separate words, in heredocs and sbatch alike.
BOARD_V2_PORTALS="portals=[{x:30.0,y_lo:10.0,y_hi:23.0},{x:60.0,y_lo:44.0,y_hi:57.0}]"

# EVAL BIN EDGES, pinned here because an eval protocol's parameters belong
# beside the numbers, not in a launcher comment. Board v1's [3,6,9,12] is WRONG
# for board v2 and would have silently unbalanced the benchmark: same-room goals
# have a 6.1cm median and crossings a 30.9cm median, so under the old edges ALL
# FOUR lower bins were pure same-room and every crossing collapsed into "12+" --
# 11 of 48 in-scope episodes, on a board that is 50/50 by construction.
# Measured over candidates; [3,8,15,25,35] gives 60 in-scope episodes at 62%
# crossing with a monotone distance spread across the whole 3-51cm range.
BOARD_V2_EVAL="eval_dist_edges=[3.0,8.0,15.0,25.0,35.0]"

BOARD_V2_PINS="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
board_w_cm=90.0 board_h_cm=60.0 min_progress_ticks=1 learning_starts=10000 \
her_n_sampled_goal=4 target_clip=10 disengaged_away_deg=60 \
require_settled=true push_cone_deg=75 same_room_goal_prob=0.5 \
push_range_min_cm=null push_range_max_cm=null object_theta_spread_deg=180 \
angular_drag_arm_cm=3.12 portal_arrival=false portal_goal=false \
portal_clearance_cm=0.5 guard_face=false rich_obs=true \
curriculum_mode=band curriculum_levels=null \
theta_tol_deg=22.5 theta_goal_window_deg=45.0 push_spawn_along_frac=null \
${BOARD_V2_PORTALS} ${BOARD_V2_EVAL}"

# The TRAINING form differs from the benchmark form in exactly one key:
# curriculum_levels. The benchmark is the reverse sampler at FULL range (null),
# which is what every arm's last curriculum level trains on; training ramps
# through 4 levels. Keeping them one string apart is what stops a scorer from
# accidentally benchmarking on level 0.
BOARD_V2_TRAIN_PINS="${BOARD_V2_PINS/curriculum_levels=null/curriculum_levels=4}"
