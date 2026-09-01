#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=recon_gamma
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aden_mckinney@seas.harvard.edu

# ===========================================================================
# v31 RECONTACT INTERFACE SWEEP -- 9 cells =
#   {recon_base, recon_goal, recon_full} x seed{0,1,2} at 1M steps.
#
# THE QUESTION. Recontact scores 0.78-0.92, but on a task that is not the one
# the graph needs: reach ONE fingertip to ONE point beside an object, from a
# free-space start. The memo's recontact moves the hand between INTERFACE
# CLASSES (Gamma_l, Eq 11) -- push, pivot, pinch -- and its target set is where
# BOTH fingertips end up. This sweep trains that and asks what it costs.
#
# WHAT CHANGES:
#
#   gamma_goal=true                        The goal becomes Eq 13's interface:
#     both fingertip targets in the OBJECT's frame plus a desired touching flag
#     for each (6-D). The object's POSE is deliberately absent -- recontact must
#     not move the object, so its pose is an input, never a target. Object-frame
#     targets are also what keeps a HER relabel valid when the object drifted
#     between the two ticks a relabel pairs.
#
#   continuous_gamma=true                  Interfaces are drawn UNIFORMLY from
#     inside a class rather than from its 4/2/8 canonical placements, which is
#     what makes Gamma_l a target SET (sec 6.1) instead of a handful of points,
#     and what makes the per-finger tolerance do real work. Pinch shares ONE
#     along-face parameter so its two contacts are DIRECTLY opposed (sec 6.3);
#     independent draws would be a torque couple, i.e. a pivot. Pivot's second
#     contact IS drawn independently -- that offset is the moment arm.
#     gamma_min_sep_cm=2.0 floors init-vs-goal separation: with continuous
#     placement an exact match has measure zero but a NEAR match is a free win
#     (the failure push_range_min_cm exists to remove). Verified min sep 2.33cm.
#
#   guard_object_still=true                Recontact's standing invariant. The
#     TARGET interface cannot be guarded -- acquiring it is the whole point --
#     but "the object does not move" must hold throughout. It used to live in
#     ContactEnv as a sticky flag folded into the arrival test; as a guard it is
#     also visible to her_valid_filter, which is the point.
#
#   her_valid_filter=true                  Relabel only TO ticks that were
#     settled and guard-valid. For recontact this is nearly FREE: arrival already
#     requires settled AND not-disturbed, so filtering the candidate pool removes
#     exactly the goals the reward would have refused to pay for. Filtering the
#     pool costs no batch size; the arrival-side version (her_settled) throws
#     away every rejected pair.
#
#   rich_obs=true                          Same observation as the push sweep:
#     contact normals in the object's frame, force, four nearest walls, xi.
#     xi's source-interface block is what distinguishes recon_full's four
#     starting modes from each other.
#
# ARMS (each adds exactly one thing):
#   recon_base   current defaults, no gamma goal. NOT redundant: the 0.78-0.92
#                numbers predate two interface changes and the benchmark pin, so
#                there is no trustworthy control on today's code. This is it.
#   recon_goal   goal Gamma in {push, pivot, pinch}, fingers still start FREE.
#   recon_full   + init Gamma in {free, push, pivot, pinch}, i.e. grasp-to-grasp,
#                which is what composition actually needs (a recontact usually
#                begins holding the push contact its predecessor left).
#
# recon_goal EXISTS TO AVOID A CONFOUND. Widening the goal set and widening the
# init set are two changes; v29's `hardmode` moved several at once and its
# verdict was uninterpretable. If recon_full drops, recon_goal says which half.
#
# MEASURED BEFORE LAUNCH (so these are not open questions):
#   Spawning into pinch/pivot does NOT kick the object: over 400 resets the
#   object moved 0.0000cm on the first tick, max. The two-finger spawn overlap
#   (radius - 0.02) was the obvious risk and it is not real.
#   Contacts present after one tick: free 0.00, mixed modes 1.23 (contact flags
#   read 0 at reset for EVERY mode -- pymunk populates them only after a step,
#   which already produced one false alarm in this project).
#
# COMPARABILITY. recon_goal/recon_full change the goal space (2-D -> 6-D), so
# SB3's check_for_correct_spaces will refuse to load a v23 checkpoint against
# them and no archived recontact number transfers. recon_base is the only arm
# comparable to history, which is the second reason it is in the sweep.
# ===========================================================================

set -e

ARMS=(recon_base recon_goal recon_full)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$(( i / 3 ))]}
SEED=${SEEDS[$(( i % 3 ))]}

# Stated explicitly in every arm, never left to a default: meta.txt's
# EXTRA_OVERRIDE is the only provenance record of what a cell actually ran.
SETTLE="eps_v_cm_s=0.5 eps_omega_deg_s=5.0"
GAMMA="gamma_goal=false mask_inactive_finger=true rich_obs=false guard_object_still=false her_valid_filter=false"

case "${ARM}" in
  recon_base) ;;                                 # base, unmodified
  # mask_inactive_finger MUST go false once the goal names BOTH fingertips:
  # masking zeroes the non-active finger's command, so half of a 6-D goal would
  # be unreachable by construction.
  recon_goal) GAMMA="gamma_goal=true continuous_gamma=true gamma_min_sep_cm=2.0 rich_obs=true mask_inactive_finger=false guard_object_still=true her_valid_filter=true goal_gamma_modes=[push,pivot,pinch] init_gamma_modes=[free]" ;;
  recon_full) GAMMA="gamma_goal=true continuous_gamma=true gamma_min_sep_cm=2.0 rich_obs=true mask_inactive_finger=false guard_object_still=true her_valid_filter=true goal_gamma_modes=[push,pivot,pinch] init_gamma_modes=[free,push,pivot,pinch]" ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

TEMPLATE="recontact"
TOTAL_STEPS=1000000
CKPT_FREQ=250000
EXTRA_OVERRIDE="use_her=true guard_terminates=true her_n_sampled_goal=4 learning_starts=10000 target_clip=10 ckpt_freq=${CKPT_FREQ} ${SETTLE} ${GAMMA}"
RUN_TAG="recon_${ARM}_s${SEED}"
GROUP="recon"

source slurm/_run_cell.sh
