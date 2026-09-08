#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=smokeC2
#SBATCH --output=logs/slurm_staging/%A_%a.out
#SBATCH --error=logs/slurm_staging/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0-03:00:00
#SBATCH --array=0-0
set -e
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/pins_board_v2.sh
TEMPLATE=push; RUN_TAG=smoke2_ctl_s0; TOTAL_STEPS=200000; SEED=0; GROUP=smokeC2
EXTRA_OVERRIDE="${BOARD_V2_TRAIN_PINS} \
obs_version=2 rich_obs=true normalize_goal_keys=false \
action_interface=contact_frame slip_model=speed_fraction slip_limit=1.0 \
gap_assist=false mask_inactive_finger=true xi_gamma_mode=face \
ckpt_freq=100000 diag_eval_episodes=32"
source slurm/_run_cell.sh
