#!/usr/bin/env bash
# Use on Hugging Face OpenEnv / training GPU pods where ~/app has no project files.
# After clone, runs start_training.sh from the real repo root.
set -euo pipefail

REPO_URL="${TRAINING_GIT_URL:-}"
TARGET_DIR="${TRAINING_HOME:-${HOME}/pricing_pro_training}"

if [[ -f "${HOME}/app/tools/run_long_training.py" ]]; then
  echo ">>> Found project in ${HOME}/app"
  exec bash "${HOME}/app/start_training.sh"
fi

if [[ -f "${TARGET_DIR}/tools/run_long_training.py" ]]; then
  echo ">>> Using existing clone: ${TARGET_DIR}"
  cd "${TARGET_DIR}"
  exec bash ./start_training.sh
fi

if [[ -z "${REPO_URL}" ]]; then
  echo "No training code in ${HOME}/app and no clone at ${TARGET_DIR}."
  echo ""
  echo "Set TRAINING_GIT_URL to your Git or Hugging Face Space git URL, then re-run:"
  echo "  export TRAINING_GIT_URL='https://github.com/ORG/REPO.git'"
  echo "  bash tools/bootstrap_training_pod.sh"
  echo ""
  echo "Or clone manually, then:"
  echo "  cd /path/to/clone && bash start_training.sh"
  exit 1
fi

echo ">>> Cloning ${REPO_URL} -> ${TARGET_DIR}"
mkdir -p "$(dirname "${TARGET_DIR}")"
if [[ -d "${TARGET_DIR}" ]]; then
  rm -rf "${TARGET_DIR}"
fi
git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
cd "${TARGET_DIR}"
exec bash ./start_training.sh
