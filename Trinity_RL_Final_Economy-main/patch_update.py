import os
import json
import torch
import re
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
MODEL_PATH = "final_pricing_pro_model" # The frozen model from the 1-day training
RLHF_FILE = "rlhf_buffer.jsonl"
MAX_PATCH_SAMPLES = 500
MAX_FEEDBACK_CHARS = 500

# Basic guardrails against obvious poisoning instructions in public feedback.
DISALLOWED_PATTERNS = [
    r"ignore previous instructions",
    r"system prompt",
    r"leak( the)? key",
    r"api key",
    r"password",
    r"exfiltrat",
    r"bypass",
]


def _sanitize_feedback(raw_feedback):
    if not isinstance(raw_feedback, str):
        return ""
    cleaned = " ".join(raw_feedback.split())
    cleaned = re.sub(r"[^\w\s.,!?-]", "", cleaned)
    cleaned = cleaned[:MAX_FEEDBACK_CHARS].strip()
    lowered = cleaned.lower()
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, lowered):
            return ""
    return cleaned


def _is_valid_entry(data):
    return (
        isinstance(data, dict)
        and isinstance(data.get("dna"), dict)
        and isinstance(data.get("action"), dict)
        and isinstance(data.get("feedback"), str)
    )

def run_patch_update():
    if not os.path.exists(RLHF_FILE):
        print("No RLHF feedback found for patching.")
        return

    print("\n>>> STARTING END-OF-DAY PATCH UPDATE (RLHF)")
    
    # 1. Load Frozen Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = 2048,
        load_in_4bit = True,
    )
    # Enable training mode for patching
    model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

    # 2. Process Feedback into Training Samples
    # We use the shopkeeper's 'Coaching' as the target behavior
    dataset = []
    with open(RLHF_FILE, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if len(dataset) >= MAX_PATCH_SAMPLES:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line {line_no} in {RLHF_FILE}")
                continue

            if not _is_valid_entry(data):
                print(f"Skipping invalid feedback schema on line {line_no}")
                continue

            sanitized_feedback = _sanitize_feedback(data.get("feedback", ""))
            if not sanitized_feedback:
                print(f"Skipping unsafe/empty feedback on line {line_no}")
                continue

            sample = {
                "instruction": f"Adjust your pricing strategy based on this shopkeeper feedback: {sanitized_feedback}",
                "input": f"Context: Customer DNA {data['dna']}. Original Prices: {data['action']}",
                "output": "Understood. I will adjust my pricing sensitivity as requested to better align with the store's goals."
            }
            dataset.append(sample)

    if not dataset:
        print("No valid RLHF samples available after validation. Skipping patch update.")
        return

    # 3. Perform Mini-Update (1-2 Epochs max)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "instruction",
        args = TrainingArguments(
            per_device_train_batch_size = 1,
            max_steps = 10, # Very fast patch
            learning_rate = 1e-5, # Low LR to avoid catastrophic forgetting
            output_dir = "patched_model_temp",
        ),
    )
    trainer.train()

    # 4. Save and Replace
    model.save_pretrained_merged(MODEL_PATH, tokenizer, save_method = "lora")
    print(">>> PATCH UPDATE COMPLETE. Model weight weights updated.")
    
    # Clear buffer after update. Refuse to remove symlinked files for safety.
    if os.path.islink(RLHF_FILE):
        print(f"Refusing to remove symlinked buffer file: {RLHF_FILE}")
    else:
        os.remove(RLHF_FILE)

if __name__ == "__main__":
    run_patch_update()
