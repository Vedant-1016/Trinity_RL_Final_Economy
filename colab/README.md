# Colab: quick 3-phase training

Open `quick_train_pricing_pro.ipynb` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**) or use **Open in Colab** if you add a Colab badge to your main README.

## What it does

1. Asks for **your** `GROQ_API_KEY` (and optional `HF_TOKEN`) — nothing is stored in the notebook.
2. Clones this repo, installs **Unsloth + TRL** and repo `requirements.txt`.
3. Generates a small SFT dataset (`SFT_SAMPLES`, default **5**) with **Groq-backed Council** when a key is set.
4. Runs **`tools/run_long_training.py --skip-sft-generation`** (same entrypoint as the [OpenEnv training Space](https://huggingface.co/spaces/Het0456/OpenEnv_training)) with **SFT 5 / heuristic 50 / council 10**, so you get `train_llm` + **plot export** + **pre_submit check** in one go. Metrics: **one row per scenario** (50 + 10).
5. Copies **`training_metrics.json`**, **`docs/*.png`**, and logs to `/content/pricing_pro_outputs/`, then zips for download. (Step 2 already produced plots; step 3 is optional re-export.)

## Requirements

- **GPU runtime** (T4 is enough for a small run; larger GPU is faster).
- Your own **Groq** key for Council / data generation in this path.

## Repo URL

In the first config cell, set `GIT_REPO_URL` to your **public** fork or upstream repo (HTTPS). Change `BRANCH` if your default branch is not `main`.

## Patches in the repo (used by the notebook)

- `generate_sft_data.py` uses explicit **`--num-samples`** (Colab cell 1 and `run_long_training.py` pass it) so a stale `SFT_SAMPLES=200` in the environment cannot override.
- `tools/run_long_training.py` is the same pipeline as `bash start_training.sh` on the Space.
