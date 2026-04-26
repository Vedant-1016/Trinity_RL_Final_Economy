#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Pricing Pro - A10G Training Launcher
# ==============================================================================
# Always run from the repo root (works from Jupyter terminal in /data, etc.)
# ==============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TRAIN_SCRIPT="$ROOT/tools/run_long_training.py"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "FATAL: Training script not found at: $TRAIN_SCRIPT"
  echo "Current directory: $ROOT"
  echo "Listing (first 40 entries):"
  ls -la "$ROOT" 2>/dev/null | head -40 || true
  echo "If this Space image is old, push this repo to the Space and click Factory rebuild."
  exit 1
fi

# 1. Set these as Hugging Face Space "Secrets" (recommended):
# - HF_TOKEN
# - GROQ_API_KEY

# 2. (Optional) Force the base model if needed
export BASE_MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# OpenEnv / bare clones often lack packages that the full Docker image installs.
if ! python -c "import dotenv" 2>/dev/null; then
  echo ">>> Installing python-dotenv (not in current environment)..."
  pip install --no-cache-dir -q "python-dotenv>=1.0.0"
fi

if [[ ! -f "$ROOT/.env" ]]; then
  if [[ -z "${GROQ_API_KEY:-}" ]]; then
    echo "Warning: no .env in repo root and GROQ_API_KEY is not set. Use Space Secrets or: export GROQ_API_KEY=..."
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Note: HF_TOKEN unset (optional for many public base models)."
  fi
fi

echo "============================================================"
echo "Pricing Pro Training (repo root: $ROOT)"
echo "Pipeline: generate_sft_data.py -> train_llm.py (SFT + RL) -> export plots -> pre_submit_check"
echo "Artifacts: final_pricing_pro_model/, training_metrics.json, docs/*.png, train_run.log"
echo "============================================================"
echo "Checking GPU..."
nvidia-smi

echo "============================================================"
echo "Starting the Python Live Tracker..."
echo "============================================================"

# SFT + Phase 2 (heuristic) + Phase 3 (council) + plots + pre-submit check
python "$TRAIN_SCRIPT" \
  --sft-samples 5 \
  --heuristic-scenarios 50 \
  --council-scenarios 10 \
  --log-file "$ROOT/train_run.log"
