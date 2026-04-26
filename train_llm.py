import os
import json
import sys
import torch
import re
import subprocess
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from persona_engine import PersonaEngine
from reviewer_engine import ReviewerEngine
from council import Council
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

def _drop_invalid_hf_tokens():
    """
    Do not call huggingface_hub.login() here: it hard-fails on bad tokens before training starts.
    Validate with whoami; if invalid, remove tokens so public model pulls can proceed anonymously.
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = (os.environ.get(key) or "").strip()
        if not token:
            continue
        try:
            from huggingface_hub import HfApi

            HfApi(token=token).whoami()
            print(f">>> Hugging Face token OK (from {key}).")
            return
        except Exception as exc:
            err_one_line = str(exc).strip().split("\n")[0][:240]
            print(
                f">>> WARNING: {key} is invalid (Hub returned error). Removing only that variable.\n"
                f">>>   {err_one_line}\n"
                ">>> Fix: https://huggingface.co/settings/tokens — then export the new token or set Space secret HF_TOKEN."
            )
            os.environ.pop(key, None)

    print(
        ">>> No valid Hugging Face token in environment — using anonymous Hub access for public models."
    )


_drop_invalid_hf_tokens()

# --- CONFIGURATION ---
MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "pricing_pro_model"
METRICS_PATH = "training_metrics.json"
SFT_SAMPLES = int(os.environ.get("SFT_SAMPLES", "5"))
HEURISTIC_SCENARIOS = int(os.environ.get("HEURISTIC_SCENARIOS", "50"))
COUNCIL_SCENARIOS = int(os.environ.get("COUNCIL_SCENARIOS", "10"))


def _atomic_write_json(path, data):
    """Write JSON atomically so a crash mid-write does not corrupt metrics."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _export_plots_subprocess():
    subprocess.run(
        [sys.executable, "tools/export_training_plots.py"],
        check=True,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
    )


# --- 1. INITIALIZE MODELS & ENGINES ---
print(">>> INITIALIZING AGENT MODEL (UNSLOTH)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# Avoid noisy generate() warning when max_new_tokens is explicitly provided.
if hasattr(model, "generation_config") and getattr(model.generation_config, "max_length", None) is not None:
    model.generation_config.max_length = None

persona_engine = PersonaEngine()
council = Council()
reviewer_engine = ReviewerEngine(council_instance=council)

# Helper to extract JSON from model output
def extract_json(text):
    if not isinstance(text, str) or not text.strip():
        return None

    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            if isinstance(obj, dict):
                candidates.append(obj)
        except json.JSONDecodeError:
            continue

    if not candidates:
        return None

    # Prefer a dict that contains the key we need for review.
    for candidate in reversed(candidates):
        if "prices" in candidate and isinstance(candidate["prices"], dict):
            return candidate

    # Fallback: return the last valid object generated.
    return candidates[-1]

# --- 2. PHASE 1: SFT BOOTSTRAP ---
def run_sft_phase():
    print(f"\n>>> STARTING PHASE 1: SFT BOOTSTRAP ({SFT_SAMPLES} SAMPLES)")
    if not os.path.exists("sft_dataset.json"):
        print("Error: sft_dataset.json not found. Run generate_sft_data.py first.")
        return
        
    with open("sft_dataset.json", "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    # Train on instruction + input + target output so the model learns
    # the exact JSON response format required during RL loops.
    dataset = []
    for sample in raw_dataset:
        instruction = sample.get("instruction", "")
        model_input = sample.get("input", "")
        model_output = sample.get("output", "")
        full_text = (
            "System: You are an AI-CFO. Output prices in JSON format.\n\n"
            f"{instruction}\n"
            f"{model_input}\n"
            f"Output: {model_output}"
        )
        dataset.append({"text": full_text})
        
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 2,
            # Small run: scale steps to dataset size (5 samples default).
            max_steps = max(3, min(30, SFT_SAMPLES * 3)),
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = OUTPUT_DIR,
        ),
    )
    trainer.train()

# --- 3. PHASE 2: ONLINE RL (HEURISTIC & COUNCIL) ---
def run_online_rl(num_scenarios=50, mode="heuristic"):
    print(f"\n>>> STARTING PHASE 2: ONLINE RL ({mode.upper()})")
    if mode == "council":
        print(
            ">>> Groq: ON for this phase (council scenarios + council_review use the Groq API if GROQ_API_KEY is set)."
        )
    else:
        print(
            ">>> Groq: OFF for heuristic phase (local math reviewer only; no Groq calls for scoring)."
        )
    phase_metrics = []
    # Exactly one JSON row per outer scenario (loops are training steps only, not extra "cases").

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for i in range(num_scenarios):
        dna = persona_engine.generate_random_dna()
        dna_dict = persona_engine.dna_to_dict(dna)
        macro_params = {"inflation": 0.05}
        products = council.generate_products(dna_dict, macro_params, min_items=3, max_items=5)
        
        # Scenario generated live
        if mode == "council":
            scenario = council.generate_scenario(dna_dict, products, macro_params)
        else:
            scenario = f"Scenario: A customer with {dna_dict['economic']} budget and {dna_dict['driver']} motivation is looking to buy."

        # THE 3-LOOP CONVERGENCE ENGINE
        current_prompt = (
            "System: You are an AI-CFO. Output prices in JSON format.\n\n"
            f"Scenario: {scenario}\n"
            f"Products: {json.dumps(products)}\n"
            "Output:"
        )

        last_row = None
        for loop in range(3):
            # A. Model Predicts Price
            inputs = tokenizer([current_prompt], return_tensors = "pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens = 200,
                do_sample = False,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.eos_token_id,
            )
            generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
            prediction = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            ai_data = extract_json(prediction)
            ai_prices = {}
            
            if not ai_data or 'prices' not in ai_data:
                success, reward, critique = False, -20.0, "Invalid JSON format. You MUST provide a JSON with the 'prices' key."
            else:
                ai_prices = ai_data['prices']
                if mode == "heuristic":
                    comp_prices = {p['name']: round(p['cost'] * 1.2, 2) for p in products}
                    success, reward, critique = reviewer_engine.heuristic_review(ai_prices, dna_dict, products, {}, comp_prices)
                else:
                    success, reward, critique = reviewer_engine.council_review(ai_prices, scenario, dna_dict, products)

            # --- LIVE BACK-PROPAGATION (ERROR-DRIVEN LEARNING) ---
            # We construct a training target that includes the feedback
            if success:
                target_output = json.dumps({"reasoning": "Optimal pricing achieved.", "prices": ai_prices})
                status_msg = f"SUCCESS (Reward: {reward})"
            else:
                # We train the model to recognize its own mistake
                target_output = json.dumps({"error": critique, "adjustment_needed": "TRUE"})
                status_msg = f"PENALTY (Reward: {reward})"

            print(f"Scenario {i} | Loop {loop+1}: {status_msg}")

            # Execute Backprop
            full_text = f"{current_prompt}\nOutput: {target_output}"
            train_inputs = tokenizer([full_text], return_tensors="pt", padding=True).to("cuda")
            train_labels = train_inputs.input_ids.clone()
            
            # Scaling the loss by the absolute value of the reward/penalty, bounded for stability
            loss_multiplier = max(min(abs(reward) / 10.0, 5.0), 0.1)
            
            outputs = model(**train_inputs, labels=train_labels)
            loss = outputs.loss * loss_multiplier
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            last_row = {
                "scenario": i,
                "loop": loop + 1,
                "mode": mode,
                "reward": float(reward),
                "loss": float(loss.detach().item()),
                "success": bool(success),
            }

            if success:
                break # EARLY EXIT
            else:
                current_prompt += f"\nCritique: {critique}. Adjust and try again."

        if last_row is not None:
            phase_metrics.append(last_row)

    return phase_metrics

# --- EXECUTION ---
# Full pipeline when run directly: SFT -> heuristic RL -> council RL -> metrics -> merged model -> plots.
# Recommended entrypoint: `python tools/run_long_training.py` (also runs generate_sft_data, plots, pre_submit).
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    model_saved_ok = False
    try:
        if not os.path.exists("sft_dataset.json"):
            from generate_sft_data import generate_sft_dataset

            generate_sft_dataset(SFT_SAMPLES)

        run_sft_phase()
        heuristic_metrics = run_online_rl(num_scenarios=HEURISTIC_SCENARIOS, mode="heuristic")
        _atomic_write_json(METRICS_PATH, heuristic_metrics)
        print(
            f">>> CHECKPOINT: heuristic phase metrics written to '{METRICS_PATH}' "
            f"({len(heuristic_metrics)} rows)"
        )

        council_metrics = run_online_rl(num_scenarios=COUNCIL_SCENARIOS, mode="council")
        all_metrics = heuristic_metrics + council_metrics
        _atomic_write_json(METRICS_PATH, all_metrics)
        print(f">>> METRICS SAVED TO '{METRICS_PATH}' ({len(all_metrics)} rows)")

        model.save_pretrained_merged("final_pricing_pro_model", tokenizer, save_method="lora")
        model_saved_ok = True
        print("\n>>> TRAINING COMPLETE. MODEL SAVED AS 'final_pricing_pro_model'")

        try:
            _export_plots_subprocess()
            print(">>> TRAINING PLOTS EXPORTED TO docs/")
        except Exception as e:
            print(f">>> WARNING: Could not export plots automatically: {e}")
    except BaseException as exc:
        print(f"\n>>> TRAINING FAILED OR INTERRUPTED: {exc}")
        raise
    finally:
        if not model_saved_ok:
            try:
                model.save_pretrained_merged(
                    "final_pricing_pro_model_interrupted",
                    tokenizer,
                    save_method="lora",
                )
                print(
                    ">>> Emergency: partial weights saved to 'final_pricing_pro_model_interrupted/' "
                    "(use if normal save never ran)"
                )
            except Exception as save_exc:
                print(f">>> Emergency model save skipped: {save_exc}")
