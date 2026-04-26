# Colab: quick 3-phase training

Open `train_colab.ipynb` in [Google Colab](https://colab.research.google.com/) (**File → Upload notebook**) or use **Open in Colab** if you add a Colab badge to your main README.

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

- `generate_sft_data.py` uses explicit **`--num-samples`** (Config + step 1, and `run_long_training.py` pass it) so a stale `SFT_SAMPLES=200` in the environment cannot override. Step 1 **asserts** the JSON row count matches `SFT_SAMPLES`.
- Shell / Space: `start_training.sh` reads **`SFT_SAMPLES`**, **`HEURISTIC_SCENARIOS`**, **`COUNCIL_SCENARIOS`** from the environment (defaults 5 / 50 / 10) so you are not locked to fixed numbers without exporting env vars.
- `train_llm.py` SFT **`max_steps`** is derived from **the number of rows in `sft_dataset.json`**, not a guessed env-only count, and is capped to avoid runaway cost on huge files.
- `tools/run_long_training.py` is the same pipeline as `bash start_training.sh` on the Space.
