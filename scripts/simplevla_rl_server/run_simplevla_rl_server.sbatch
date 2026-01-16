#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:h200:1
#SBATCH --job-name=simplevla-rl-server
#SBATCH --output=logs/simplevla_server_%j.out
#SBATCH --error=logs/simplevla_server_%j.err
#SBATCH --time=06:00:00
#SBATCH --comment="preemption=yes;requeue=true"

# Exit on error
set -e

echo "========================================"
echo "SimpleVLA-RL Policy Server - HPC Setup"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Checkpoint paths
RL_CHECKPOINT="/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/SimpleVLA-RL/openvla_worldgym_bridge_oft_ckpt20k_5chunk/actor/global_step_64"
# RL_CHECKPOINT="/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/openvla-oft-bridge-sft-20000"
# RL_CHECKPOINT="/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/SimpleVLA-RL/bridge_worldgym_bridge_oft_ckpt20k_5chunk/actor/global_step_24"
SFT_CHECKPOINT="/projects/work/yang-lab2/as20482/checkpoints/SimpleVLA-RL/openvla-oft-bridge-sft-20000"

# Server configuration
SERVER_PORT=8000
AUTO_EVAL_DIR="/projects/work/yang-lab2/as20482/auto_eval"

# Create logs directory
mkdir -p "${AUTO_EVAL_DIR}/logs"

# Change to project directory
cd "${AUTO_EVAL_DIR}"

# Run startup script (handles environment and tunnel)
bash scripts/simplevla_rl_server/start_simplevla_rl_server.sh \
    "${RL_CHECKPOINT}" \
    "${SFT_CHECKPOINT}" \
    "${SERVER_PORT}"

echo ""
echo "========================================"
echo "Server stopped at $(date)"
echo "========================================"
