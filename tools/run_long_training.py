import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


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
    parser.add_argument("--sft-samples", type=int, default=200)
    parser.add_argument("--heuristic-scenarios", type=int, default=4000)
    parser.add_argument("--council-scenarios", type=int, default=500)
    parser.add_argument("--skip-sft-generation", action="store_true")
    parser.add_argument("--log-file", default="train_run.log")
    args = parser.parse_args()

    if args.sft_samples <= 0 or args.heuristic_scenarios <= 0 or args.council_scenarios <= 0:
        raise ValueError("All scenario/sample arguments must be positive integers.")

    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY is not set. Add it to .env or environment before training.")
    if not os.environ.get("HF_TOKEN"):
        print(">>> WARNING: HF_TOKEN not set. Private/gated model pulls may fail.")

    try:
        import torch  # type: ignore
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not has_cuda:
        raise EnvironmentError(
            "CUDA GPU not detected in this runtime. Run this script on your A10G/HF GPU runtime."
        )

    os.makedirs("docs", exist_ok=True)
    env = os.environ.copy()
    env["SFT_SAMPLES"] = str(args.sft_samples)
    env["HEURISTIC_SCENARIOS"] = str(args.heuristic_scenarios)
    env["COUNCIL_SCENARIOS"] = str(args.council_scenarios)

    estimated_minutes = _estimate_time_minutes(
        args.heuristic_scenarios, args.council_scenarios
    )
    print(">>> FULL TRAINING RUN CONFIG")
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
            _run_and_stream(
                [sys.executable, "generate_sft_data.py"],
                env=env,
                log_file_path=args.log_file,
            )

        _run_and_stream(
            [sys.executable, "train_llm.py"],
            env=env,
            log_file_path=args.log_file,
        )

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
