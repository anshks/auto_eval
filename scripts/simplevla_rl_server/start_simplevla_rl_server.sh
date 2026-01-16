#!/bin/bash
set -e

# Parse arguments
RL_CHECKPOINT="$1"
SFT_CHECKPOINT="$2"
SERVER_PORT="${3:-8000}"

echo "========================================"
echo "Starting SimpleVLA-RL Server"
echo "========================================"
echo "RL Checkpoint: ${RL_CHECKPOINT}"
echo "SFT Checkpoint: ${SFT_CHECKPOINT}"
echo "Server Port: ${SERVER_PORT}"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate autoeval

# GPU and environment settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export ROBOT_PLATFORM=WORLDGYM

# Verify checkpoints exist
if [ ! -d "${RL_CHECKPOINT}" ]; then
    echo "ERROR: RL checkpoint not found at ${RL_CHECKPOINT}"
    exit 1
fi

if [ ! -d "${SFT_CHECKPOINT}" ]; then
    echo "ERROR: SFT checkpoint not found at ${SFT_CHECKPOINT}"
    exit 1
fi

echo ""
echo "Environment ready. Starting bore.pub tunnel..."

# Create logs directory
mkdir -p logs

# Start bore.pub tunnel in background
bore local ${SERVER_PORT} --to bore.pub > logs/bore_simplevla_${SLURM_JOB_ID}.log 2>&1 &
BORE_PID=$!

# Wait a bit for tunnel to establish
sleep 5

# Extract tunnel URL from bore output
echo ""
echo "========================================"
echo "Bore Tunnel Status"
echo "========================================"
cat logs/bore_simplevla_${SLURM_JOB_ID}.log
echo "========================================"
echo ""

# Save connection info to file
mkdir -p server_info
echo "bore.pub" > server_info/current_server_ip.txt
grep -oP "bore.pub:\K\d+" logs/bore_simplevla_${SLURM_JOB_ID}.log | head -1 > server_info/current_server_port.txt
echo "${SLURM_JOB_ID}" > server_info/current_job_id.txt
date > server_info/last_started.txt

echo "========================================"
echo "Starting FastAPI Server"
echo "========================================"
echo "Loading SimpleVLA-RL policy (this may take a few minutes)..."
echo ""

# Start FastAPI server (foreground, will block)
python auto_eval/policy_server/simplevla_rl_server.py \
    --rl_checkpoint "${RL_CHECKPOINT}" \
    --sft_checkpoint "${SFT_CHECKPOINT}" \
    --host "0.0.0.0" \
    --port ${SERVER_PORT}

# Cleanup on exit
echo ""
echo "Server shutting down, cleaning up tunnel..."
kill $BORE_PID 2>/dev/null || true
echo "Cleanup complete."
