# Colab: quick 3-phase training

Open `quick_train_pricing_pro.ipynb` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**) or use **Open in Colab** if you add a Colab badge to your main README.

## What it does

1. Asks for **your** `GROQ_API_KEY` (and optional `HF_TOKEN`) — nothing is stored in the notebook.
2. Clones this repo, installs **Unsloth + TRL** and repo `requirements.txt`.
3. Generates a small SFT dataset (`SFT_SAMPLES`, default **5**) with **Groq-backed Council** when a key is set.
4. Runs `train_llm.py` with **`HEURISTIC_SCENARIOS=50`** and **`COUNCIL_SCENARIOS=10`** (same as repo / `run_long_training` defaults) so all **three** phases run. Metrics: **one row per scenario** (50 + 10), not per inner loop.
5. Runs `tools/export_training_plots.py` and copies **`training_metrics.json`**, **`docs/*.png`**, and the training log to `/content/pricing_pro_outputs/`, then zips it for download.

## Requirements

- **GPU runtime** (T4 is enough for a small run; larger GPU is faster).
- Your own **Groq** key for Council / data generation in this path.

## Repo URL

In the first config cell, set `GIT_REPO_URL` to your **public** fork or upstream repo (HTTPS). Change `BRANCH` if your default branch is not `main`.

## Patches in the repo (used by the notebook)

- `generate_sft_data.py` reads default `--num-samples` from the **`SFT_SAMPLES`** environment variable so orchestration and Colab stay aligned.
- `train_llm.py` includes `import sys` for plot export.
