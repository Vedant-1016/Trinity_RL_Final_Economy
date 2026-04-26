import argparse
import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fresh clones / bare pods may not have python-dotenv; secrets still work via os.environ.
    pass


def _run_and_stream(command, env, log_file_path):
    print(f"\n>>> RUNNING: {' '.join(command)}")
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n\n===== COMMAND START {datetime.utcnow().isoformat()}Z =====\n")
        log_file.write(" ".join(command) + "\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        log_file.write(f"\n===== COMMAND END (exit={return_code}) =====\n")
        log_file.flush()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_training_stack() -> None:
    """Install Unsloth/TRL on first use (e.g. user runs this script; start_training.sh also installs)."""
    try:
        import unsloth  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError):
        pass
    root = _repo_root()
    req = root / "requirements-training.txt"
    if not req.is_file():
        raise EnvironmentError(
            "unsloth is not installed, and requirements-training.txt is missing at the repo root.\n"
            "From /home/user/app run:\n"
            "  pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install --no-cache-dir unsloth 'trl>=0.9' transformers datasets accelerate peft bitsandbytes safetensors"
        ) from None
    print(">>> unsloth not found; installing CUDA PyTorch + requirements-training.txt (first run, a few minutes)...", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        env=os.environ.copy(),
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(req)],
        env=os.environ.copy(),
    )
    importlib.invalidate_caches()
    import unsloth  # noqa: F401
    print(">>> unsloth import OK", flush=True)


def _estimate_time_minutes(heuristic_scenarios, council_scenarios):
    # Conservative planning estimate on A10G:
    # heuristic scenario: ~2.0 sec average
    # council scenario: ~5.0 sec average (extra API calls)
    estimated_seconds = (heuristic_scenarios * 2.0) + (council_scenarios * 5.0) + (25 * 60)
    return round(estimated_seconds / 60.0, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Run full SFT + online training in one go with live logs."
    )
    parser.add_argument("--sft-samples", type=int, default=5)
    parser.add_argument("--heuristic-scenarios", type=int, default=50)
    parser.add_argument("--council-scenarios", type=int, default=10)
    parser.add_argument("--skip-sft-generation", action="store_true")
    parser.add_argument("--log-file", default="train_run.log")
    args = parser.parse_args()

    if args.sft_samples <= 0 or args.heuristic_scenarios <= 0 or args.council_scenarios <= 0:
        raise ValueError("All scenario/sample arguments must be positive integers.")

    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to .env or environment before training.")
    if not os.environ.get("HF_TOKEN"):
        print(">>> WARNING: HF_TOKEN not set. Private/gated model pulls may fail.")

    # train_llm.py needs unsloth+trl; this script is often run without bash start_training.sh
    if os.path.isfile(_repo_root() / "train_llm.py"):
        _ensure_training_stack()

    torch_import_error: Optional[Exception] = None
    has_cuda = False
    try:
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        torch_import_error = exc

    if not has_cuda:
        details = []
        if torch_import_error is not None:
            details.append(f"PyTorch import failed: {torch_import_error!r}")
        else:
            try:
                import torch  # type: ignore

                details.append(
                    f"PyTorch {torch.__version__}  bundled_cuda={getattr(torch.version, 'cuda', None)!r}  "
                    f"is_available={torch.cuda.is_available()}"
                )
            except Exception:  # noqa: BLE001
                pass
        details.append(
            "The GPU driver can work (nvidia-smi) while this Python has no usable CUDA in PyTorch "
            "(CPU-only wheel, missing install, or wrong interpreter)."
        )
        details.append(
            "Example fix in this same terminal: install a CUDA build, then training deps, e.g.\n"
            "  pip install -U 'torch' --index-url https://download.pytorch.org/whl/cu124\n"
            "  pip install -U 'xformers' 'bitsandbytes'  # as needed for unsloth/trl"
        )
        raise EnvironmentError(
            "CUDA GPU not available to this Python (torch). "
            "Run on a GPU host and use a PyTorch build with CUDA.\n"
            + "\n".join(details)
        ) from torch_import_error

    os.makedirs("docs", exist_ok=True)
    env = os.environ.copy()
    env["SFT_SAMPLES"] = str(args.sft_samples)
    env["HEURISTIC_SCENARIOS"] = str(args.heuristic_scenarios)
    env["COUNCIL_SCENARIOS"] = str(args.council_scenarios)

    estimated_minutes = _estimate_time_minutes(
        args.heuristic_scenarios, args.council_scenarios
    )
    print(">>> FULL TRAINING RUN CONFIG")
    print(
        f"- Groq key set: {bool(os.environ.get('GROQ_API_KEY'))} "
        f"(SFT data uses Groq when key present; heuristic RL does not use Groq)"
    )
    print(f"- SFT samples: {args.sft_samples}")
    print(f"- Heuristic scenarios: {args.heuristic_scenarios}")
    print(f"- Council scenarios: {args.council_scenarios}")
    print(f"- Log file: {args.log_file}")
    print(f"- Estimated runtime (A10G): ~{estimated_minutes} minutes")
    print(f"- Base model: {env.get('BASE_MODEL_NAME', 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit')}")
    print("- Council synthesizer model: gemma2-9b-it")
    print("- Live logs stream below and are also written to the log file.")

    start = time.time()
    try:
        if not args.skip_sft_generation:
            # Pass --num-samples on the CLI so a stale SFT_SAMPLES in the parent shell cannot override.
            _run_and_stream(
                [
                    sys.executable,
                    "generate_sft_data.py",
                    "--num-samples",
                    str(args.sft_samples),
                ],
                env=env,
                log_file_path=args.log_file,
            )

        try:
            _run_and_stream(
                [sys.executable, "train_llm.py"],
                env=env,
                log_file_path=args.log_file,
            )
        except RuntimeError:
            # If training stopped early, still generate plots from any metrics on disk.
            metrics_path = "training_metrics.json"
            if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 8:
                print(
                    "\n>>> Training subprocess failed; attempting plot export from existing metrics..."
                )
                try:
                    _run_and_stream(
                        [sys.executable, "tools/export_training_plots.py"],
                        env=env,
                        log_file_path=args.log_file,
                    )
                except RuntimeError as plot_exc:
                    print(f">>> Plot export after partial training skipped: {plot_exc}")
            raise

        _run_and_stream(
            [sys.executable, "tools/export_training_plots.py"],
            env=env,
            log_file_path=args.log_file,
        )

        _run_and_stream(
            [sys.executable, "tools/pre_submit_check.py"],
            env=env,
            log_file_path=args.log_file,
        )
    except Exception as exc:
        elapsed = round((time.time() - start) / 60.0, 1)
        print(f"\n>>> TRAINING PIPELINE FAILED after {elapsed} minutes: {exc}")
        raise

    elapsed = round((time.time() - start) / 60.0, 1)
    print("\n>>> TRAINING PIPELINE COMPLETE")
    print(f"- Elapsed: {elapsed} minutes")
    print("- Model output: final_pricing_pro_model/")
    print("- Metrics: training_metrics.json")
    print("- Plots: docs/reward_curve.png, docs/loss_curve.png, docs/baseline_vs_trained.png")
    print(f"- Log file: {args.log_file}")


if __name__ == "__main__":
    main()
