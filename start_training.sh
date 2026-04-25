#!/bin/bash

# ==============================================================================
# Pricing Pro - A10G Training Launcher
# ==============================================================================
# This script sets up your environment variables and runs the full live-tracking
# training suite (tools/run_long_training.py) which your friend recently added!
# ==============================================================================

# 1. Set these as Hugging Face Space "Secrets" (recommended):
# - HF_TOKEN
# - GROQ_API_KEY

# 2. (Optional) Force the base model if needed
export BASE_MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

echo "============================================================"
echo "🚀 Starting Pricing Pro Training on A10G"
echo "============================================================"
echo "Checking GPU..."
nvidia-smi

echo "============================================================"
echo "Starting the Python Live Tracker..."
echo "============================================================"

# This runs the wrapper script your friend made, which handles SFT generation,
# Phase 1, Phase 2, Phase 3, and automatically plots the graphs!
# It will stream the output live to your console and save it to train_run.log
python tools/run_long_training.py --sft-samples 200 --heuristic-scenarios 4000 --council-scenarios 500
