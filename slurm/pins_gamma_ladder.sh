#!/bin/bash
# slurm/pins_gamma_ladder.sh -- THE one definition of the Gamma ladder's task.
#
# SEPARATE FILE FROM pins_board_v2.sh ON PURPOSE. Recontact has no board and no
# portals (a locked scope decision, docs/stage1_env_spec.md), and ContactEnv
# REFUSES push-only keys rather than ignoring them -- theta_tol_deg,
# push_cone_deg, same_room_goal_prob, require_settled and portal_* all raise on
# recontact. Sourcing board v2's pins here is a hard error, which is the correct
# behaviour and is why these are two files rather than one string with branches.
#
# ---------------------------------------------------------------------------
# WHAT THIS TASK IS
#
# Recontact's job is to move the fingers between CANONICAL CONTACT INTERFACES
# while the object stays put: the memo's node is v = (C_i, Theta_j, Gamma_l), and
# recontact changes Gamma while holding C and Theta fixed. Changing the
# orientation bin is `pivot`'s job, not this one.
#
#   gamma_goal=count (D1).
#     Arrival is the CONTACT COUNT plus the object being settled, with NO
#     positional tolerance. The 6-D goal vector is unchanged, so the wire format
#     and every archived observation Box survive; the drawn fingertip positions
#     are one member of the target set (sec 6.1) and reaching any member counts.
#     Relabel-safe by construction: the rule reads both counts off the goal
#     vectors, so a relabeled goal is graded by exactly the rule its own achieved
#     state satisfies -- no per-transition field to forget, which is what made
#     63 GPU-hours of the previous Gamma arms uninterpretable.
#
#   init_gamma_modes -- ARBITRARY START, but the COUNT MUST CHANGE.
#     The default is ("free",) alone, i.e. acquisition from free space, which is
#     not the experiment: composition needs grasp-to-grasp (push -> recontact ->
#     pinch starts holding a push contact). So the start is drawn from the other
#     classes too.
#
#     BUT: under a count goal, a start whose count ALREADY EQUALS the goal's is
#     satisfied at t=0. Measured, and this is why the arms below exclude such
#     pairs -- with init drawn from all four classes the UNTRAINED floors were
#     g_one 0.583 and g_two_disp 0.396, because roughly half of episodes spawned
#     already holding the commanded number of contacts. An arm with a 0.58 floor
#     measures nothing.
#
#     THIS IS A REAL LIMITATION OF THE COUNT ABSTRACTION AND IS REPORTED, NOT
#     HIDDEN: pinch <-> pivot is a genuine interface transition (two contacts,
#     different geometry) that contact count cannot express at all, so it reads
#     as a no-op. Contact count buys transfer to a T-shape, a round object and
#     3D; it pays for it by collapsing same-count regrasps. If those matter, they
#     need the positional Gamma, which is a separate (and currently 0.000) arm.
#
#   guard_object_still=displacement, eps=2cm (D2), v_max_cm_s=2, horizon=400.
#     All four are one hypothesis, and the probe says they are complementary
#     rather than alternatives. The velocity guard is an instantaneous 0.5cm/s
#     test that fired on 237 of 240 scripted episodes at a median tick 8 of 200,
#     and -- because the flag is STICKY and gates _her_arrived -- it had switched
#     HER OFF for ~96% of every episode. At a 2cm/s action scale the peak
#     disturbance is a median 0.62-0.76cm (p90 1.03-1.19), so eps=2cm leaves
#     1.000 of ticks eligible; at 20cm/s even eps=3cm leaves only 0.33-0.43.
#     horizon=400 because v_max=2 x horizon=200 x dt=0.04 gives only 16cm of
#     finger travel against a disengaged spawn radius of 8.0-16.1cm -- not enough
#     for two sequential placements, and the previous arm returned horizon-exit
#     for a reason unrelated to its hypothesis.
#
#   continuous_gamma=true -- draw the interface continuously from its class, so
#     Gamma_l is a target SET (sec 6.1) rather than a handful of canonical points.
#
# NO SUCCESS BAR. This is a measurement. But 0.425 is what a crude scripted
# controller achieves on one contact, so g_one below that is a LEARNING failure,
# not a task failure. g_two_disp has no scripted reference -- the controller ran
# out of horizon -- so it measures rather than compares.
# ---------------------------------------------------------------------------

GAMMA_PINS="use_her=true w_d=0 w_a=0 w_F=0 w_m=0 w_T=0 guard_terminates=true \
min_progress_ticks=1 learning_starts=10000 her_n_sampled_goal=4 target_clip=10 \
disengaged_away_deg=60 angular_drag_arm_cm=3.12 \
gamma_goal=count continuous_gamma=true gamma_min_sep_cm=2.0 \
guard_object_still=displacement guard_disp_eps_cm=2.0 \
v_max_cm_s=2.0 horizon=400 rich_obs=true"

# Bash brace-expands [{a,b}] and also [a,b] in some contexts, so every Hydra list
# override reaches the CLI through a variable.
# THE LADDER IS DIRECTIONAL, and the floors are why. Excluding same-count pairs
# was not enough: with init=[free,pivot,pinch] the untrained g_one floor was
# still 0.542, because two thirds of those episodes START with two contacts and
# a random policy reaches "one contact" by simply DRIFTING OFF one of them.
#
# MEASURED FINDING, recorded rather than hidden: RELEASING a contact is not a
# skill. Its untrained floor is 0.542. ACQUIRING one is. So the ladder rungs are
# acquisitions, and the release direction is reported as the 0.542 column it is.
#
# g_one: acquire ONE contact from free space. Floor 0.000.
GAMMA_GOAL_ONE="goal_gamma_modes=[push]"
GAMMA_INIT_ONE="init_gamma_modes=[free]"

# g_two_disp: acquire TWO contacts, from free space OR from a single existing
# push contact -- the grasp-to-grasp case composition actually needs. This is the
# arm that tests whether the second contact is acquirable at all once the guard
# stops killing the episode. Floor 0.000.
GAMMA_GOAL_TWO="goal_gamma_modes=[pivot,pinch]"
GAMMA_INIT_TWO="init_gamma_modes=[free,push]"
