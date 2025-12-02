#!/bin/bash

# Training script for Franka dataset with Pi05

# Proxy settings for faster downloads (uncomment and configure if you have a proxy)
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port

# JAX environment variables
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_$USER
export TMPDIR=/tmp
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

# Create cache directory if it doesn't exist
mkdir -p $JAX_COMPILATION_CACHE_DIR

# Create logs directory
mkdir -p logs

# Run training in background with nohup
nohup uv run scripts/train.py pi05_franka \
    --exp-name=franka_peg_in_hole \
    --overwrite \
    "$@" > logs/train_franka_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID and show log info
PID=$!
echo "Training started in background with PID: $PID"
echo "Log file: logs/train_franka_$(date +%Y%m%d_%H%M%S).log"
echo "To monitor progress: tail -f logs/train_franka_*.log"
echo "To stop training: kill $PID"
