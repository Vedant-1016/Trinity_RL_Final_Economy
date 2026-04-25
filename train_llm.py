import os
import json
import torch
import re
import subprocess
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from persona_engine import PersonaEngine
from reviewer_engine import ReviewerEngine
from council import Council
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()

# Authenticate with Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# --- CONFIGURATION ---
MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "pricing_pro_model"
METRICS_PATH = "training_metrics.json"
SFT_SAMPLES = int(os.environ.get("SFT_SAMPLES", "200"))
HEURISTIC_SCENARIOS = int(os.environ.get("HEURISTIC_SCENARIOS", "2000"))
COUNCIL_SCENARIOS = int(os.environ.get("COUNCIL_SCENARIOS", "25"))

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
    return None

# --- 2. PHASE 1: SFT BOOTSTRAP (200 SAMPLES) ---
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
            warmup_steps = 5,
            max_steps = 60, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            output_dir = OUTPUT_DIR,
        ),
    )
    trainer.train()

# --- 3. PHASE 2: ONLINE RL (HEURISTIC & COUNCIL) ---
def run_online_rl(num_scenarios=4000, mode="heuristic"):
    print(f"\n>>> STARTING PHASE 2: ONLINE RL ({mode.upper()})")
    phase_metrics = []

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

            phase_metrics.append({
                "scenario": i,
                "loop": loop + 1,
                "mode": mode,
                "reward": float(reward),
                "loss": float(loss.detach().item()),
                "success": bool(success),
            })

            if success:
                break # EARLY EXIT
            else:
                current_prompt += f"\nCritique: {critique}. Adjust and try again."

    return phase_metrics

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure SFT data exists
    if not os.path.exists("sft_dataset.json"):
        from generate_sft_data import generate_sft_dataset
        generate_sft_dataset(SFT_SAMPLES)

    run_sft_phase()
    heuristic_metrics = run_online_rl(num_scenarios=HEURISTIC_SCENARIOS, mode="heuristic")
    council_metrics = run_online_rl(num_scenarios=COUNCIL_SCENARIOS, mode="council")

    all_metrics = heuristic_metrics + council_metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f">>> METRICS SAVED TO '{METRICS_PATH}'")
    
    model.save_pretrained_merged("final_pricing_pro_model", tokenizer, save_method = "lora")
    print("\n>>> TRAINING COMPLETE. MODEL SAVED AS 'final_pricing_pro_model'")

    try:
        subprocess.run(["python", "tools/export_training_plots.py"], check=True)
        print(">>> TRAINING PLOTS EXPORTED TO docs/")
    except Exception as e:
        print(f">>> WARNING: Could not export plots automatically: {e}")
