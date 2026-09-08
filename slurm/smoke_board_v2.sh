#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=smokeBv2
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-03:00:00
#SBATCH --array=0-1

# 200k SMOKE ON BOARD V2 -- the 1.5h check that stands between a task-design
# mistake and ~220 GPU-hours of Sweep C.
#
# Two cells, both seed 0: `ctl` (the arm Bar 1's shape is read on) and `g_one`
# (the Gamma ladder's easier rung, which also exercises the displacement guard,
# v_max=2 and horizon=400 end to end). Not a result -- a liveness check.
#
# WHAT IT MUST SHOW:
#   ctl     rollout/success_rate climbs off its 0.000 floor, and the sampler's
#           curriculum_leaks stays 0. A leak falls through to the FORWARD sampler
#           at the wrong level distribution, i.e. silently trains a different task.
#   g_one   the same, plus the Gamma TRAVEL BUDGET holds: v_max=2 x horizon=400 x
#           dt=0.04 = 32cm of finger travel against a disengaged spawn radius of
#           8.0-16.1cm. If episodes end on horizon with the fingers still far
#           from the object, the budget is wrong and the arm would have returned
#           0.000 for a reason unrelated to its hypothesis.
set -e
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source slurm/pins_board_v2.sh
source slurm/pins_gamma_ladder.sh

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  TEMPLATE=push; RUN_TAG=smoke_ctl_s0
  EXTRA_OVERRIDE="${BOARD_V2_TRAIN_PINS} \
obs_version=2 rich_obs=true normalize_goal_keys=false \
action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
gap_assist=false mask_inactive_finger=true xi_gamma_mode=face \
ckpt_freq=100000 diag_eval_episodes=32"
else
  TEMPLATE=recontact; RUN_TAG=smoke_g_one_s0
  EXTRA_OVERRIDE="${GAMMA_PINS} \
obs_version=2 rich_obs=true normalize_goal_keys=false xi_gamma_mode=count \
action_interface=finger_velocity slip_model=speed_fraction slip_limit=1.0 \
mask_inactive_finger=false gap_assist=false \
${GAMMA_GOAL_ONE} ${GAMMA_INIT_ONE} \
ckpt_freq=100000 diag_eval_episodes=32"
fi

TOTAL_STEPS=200000
SEED=0
GROUP=smokeBv2
source slurm/_run_cell.sh
