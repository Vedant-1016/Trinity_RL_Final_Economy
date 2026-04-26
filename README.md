---
title: OpenEnv Training
emoji: "💻🐳"
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
tags:
  - jupyterlab
suggested_storage: small
---

## Pricing Pro (Training Space)

This Space is set up for **GPU training + JupyterLab** for the Pricing Pro / OpenEnv project.

### Link this repo to the Space (one-time)

1. In Hugging Face: open this Space → **Settings** → confirm the **Repository** is the same Git repo that contains `tools/run_long_training.py`.
2. After every push that should change training code: **Settings → Factory rebuild** (or trigger a new build) so the Docker image includes the latest files.

### Run training (recommended)

You do **not** need to run `train_llm.py` by itself for a full Space run: `tools/run_long_training.py` already runs **`generate_sft_data.py` → `train_llm.py` (SFT + heuristic + council) → plot export → pre-submit check**. Use `bash start_training.sh` or `python tools/run_long_training.py ...`.

JupyterLab’s terminal often starts in **`/data`**, not the app folder. The repo code is in **`/home/user/app`**.

Default run sizes in `start_training.sh` / `tools/run_long_training.py` are **SFT=5**, **heuristic=50**, **council=10**; `training_metrics.json` has **one row per outer scenario** (50 + 10, not 50×3). Override with `--sft-samples`, `--heuristic-scenarios`, `--council-scenarios` or the same env var names.

Use either:

```bash
bash /home/user/app/start_training.sh
```

or:

```bash
cd /data/pricing_pro_app
bash start_training.sh
```

(`/data/pricing_pro_app` is a symlink to `/home/user/app`; see `/data/WHERE_IS_TRAINING.txt`.)

### Secrets

In the Space: **Settings → Secrets and variables → Secrets** (or **Repository secrets**), add exactly:

- **`HF_TOKEN`** — a valid Hugging Face access token (read is enough for public models; write is fine).
- **`GROQ_API_KEY`** — your Groq API key.

Names are **case-sensitive** and must match those strings so the runtime exports them as environment variables.

**Important:** those values are available to processes running **inside that Space’s container** (e.g. Jupyter terminal on `/home/user/app`). They are **not** automatically injected into a **separate** OpenEnv training pod or into a shell where you only `git clone` the repo elsewhere. For a clone-on-bare-GPU workflow, run `export HF_TOKEN=...` and `export GROQ_API_KEY=...` in that terminal (or use a `.env` file — never commit it).

If you still see **401 Invalid user token**, the running environment is using a different or stale `HF_TOKEN` than the one you expect: rotate the token in [HF token settings](https://huggingface.co/settings/tokens), update the Space secret, restart the Space / pod, and avoid exporting an old token in `~/.bashrc`.

### If you see `No such file or directory` for `tools/run_long_training.py`

**Case A — Docker Space (this repo’s `Dockerfile`):** the running image is old or not built from this repo. Push the latest commit, then **Settings → Factory rebuild** the Space.

**Case B — OpenEnv / training GPU pod (`r-...-openenv-training-...`):** those pods often ship with an **empty `~/app`**. They are **not** the same filesystem as a rebuilt Space image. You must **clone the git repo once**, then train from the clone.

Diagnostics (paste in the pod terminal):

```bash
pwd
ls -la ~
ls -la ~/app 2>/dev/null || true
ls -la /home/user/app 2>/dev/null || true
```

If `~/app` has no `start_training.sh`, clone the **real** repo URL (not a placeholder). This project’s upstream is public:

```bash
# If git is waiting at "Username for 'https://github.com':" press Ctrl+C first.
rm -rf ~/pricing_pro_training
git clone --depth 1 https://github.com/Vedant-1016/Trinity_RL_Final_Economy.git ~/pricing_pro_training
cd ~/pricing_pro_training
chmod +x start_training.sh tools/bootstrap_training_pod.sh 2>/dev/null || true
# Bare GPU pods (not the Space Docker image) need Python deps once:
pip install --no-cache-dir -r requirements.txt
bash start_training.sh
```

If you use a **fork**, replace the URL with the HTTPS URL from your fork’s green **Code** button on GitHub.

Private repo: use an HTTPS URL with a read token, or `git clone` after `huggingface-cli login` / SSH key setup.

Optional bootstrap (same public repo):

```bash
export TRAINING_GIT_URL="https://github.com/Vedant-1016/Trinity_RL_Final_Economy.git"
curl -fsSL "https://raw.githubusercontent.com/Vedant-1016/Trinity_RL_Final_Economy/main/tools/bootstrap_training_pod.sh" -o /tmp/bootstrap_training_pod.sh
bash /tmp/bootstrap_training_pod.sh
```

### OpenEnv deliverables (links)

- [Hugging Face training Space (Docker + JupyterLab)](https://huggingface.co/spaces) — set this to your live Space URL after you publish; it must match the repo wired in Space settings.
- [Project writeup (GitHub repository)](https://github.com/Vedant-1016/Trinity_RL_Final_Economy) — add a short blog, video, or slide deck URL here when you have one, or use this repo for the written report.

Training notebook (also required on disk for checks): `colab/quick_train_pricing_pro.ipynb` (in-repo path; open in Colab from GitHub with **File → Open notebook** on that path if the repo is public).
