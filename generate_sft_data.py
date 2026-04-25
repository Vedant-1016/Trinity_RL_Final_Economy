import json
import random
import os
import argparse
from persona_engine import PersonaEngine
from council import Council

def _offline_scenario(dna_dict, products, macro_params):
    inflation = macro_params.get("inflation")
    sentiment = macro_params.get("sentiment")
    basket = ", ".join(p["name"].replace("_", " ") for p in products)
    return (
        f"A shopper with a {dna_dict['economic']} budget and a {dna_dict['driver']} motivation is comparing a basket of "
        f"{basket}. Inflation is running at {inflation:.2f} and consumer sentiment is {sentiment:.2f}. "
        f"They are a {dna_dict['intelligence']} with {dna_dict['urgency']} urgency buying for a {dna_dict['scale']} context—"
        f"they will punish prices that feel unfair but will pay for clear value."
    )


def _offline_margin(dna_dict, macro_params):
    inflation = float(macro_params.get("inflation", 0.05))
    sentiment = float(macro_params.get("sentiment", 0.6))
    margin = 0.22 + (sentiment - 0.6) * 0.08 - (inflation - 0.05) * 0.10

    driver = dna_dict.get("driver")
    intelligence = dna_dict.get("intelligence")
    urgency = dna_dict.get("urgency")
    economic = dna_dict.get("economic")

    if driver == "Status":
        margin += 0.08
    elif driver == "Performance":
        margin += 0.04
    elif driver == "Frugality":
        margin -= 0.06

    if intelligence == "Loyalist":
        margin += 0.03
    elif intelligence == "Researcher":
        margin -= 0.03

    if urgency == "Panic-Buyer":
        margin += 0.05

    if economic == "Fixed/Low":
        margin -= 0.05
    elif economic == "High":
        margin += 0.03

    return max(0.08, min(margin, 0.45))


def generate_sft_dataset(num_samples=200, out_path="sft_dataset.json", seed=None, offline=False):
    engine = PersonaEngine()
    has_groq_key = bool(os.environ.get("GROQ_API_KEY"))
    use_offline = offline or not has_groq_key
    council = None if use_offline else Council()
    dataset = []

    if seed is not None:
        random.seed(seed)

    mode = "OFFLINE" if use_offline else "GROQ"
    print(f"Generating {num_samples} SFT samples ({mode})...")

    for i in range(num_samples):
        dna = engine.generate_random_dna()
        dna_dict = engine.dna_to_dict(dna)
        
        # Randomize Macro Params for variety
        macro_params = {
            "inflation": round(random.uniform(0.01, 0.15), 2),
            "sentiment": round(random.uniform(0.1, 1.0), 2)
        }
        if use_offline:
            from product_catalog import sample_products
            products = sample_products(min_items=3, max_items=5)
            scenario = _offline_scenario(dna_dict, products, macro_params)
            margin = _offline_margin(dna_dict, macro_params)
            golden_prices = {p["name"]: round(float(p["cost"]) * (1.0 + margin), 2) for p in products}
            reasoning = f"Offline heuristic margin={margin:.2f} applied to costs."
        else:
            products = council.generate_products(dna_dict, macro_params, min_items=3, max_items=5)
            scenario = council.generate_scenario(dna_dict, products, macro_params)
            golden_prices = council.generate_golden_price(dna_dict, products, macro_params)
            reasoning = "Determined by council logic."

        sample = {
            "instruction": f"Set prices for the following products based on the market scenario.\n\nScenario: {scenario}",
            "input": f"Products: {json.dumps(products)}\nMarket Conditions: {json.dumps(macro_params)}",
            "output": json.dumps({"reasoning": reasoning, "prices": golden_prices}, indent=2)
        }
        dataset.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Successfully saved {num_samples} samples to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Pricing Pro.")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--out", type=str, default="sft_dataset.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--offline", action="store_true", help="Disable Groq calls and use heuristic generation.")
    args = parser.parse_args()

    generate_sft_dataset(
        num_samples=args.num_samples,
        out_path=args.out,
        seed=args.seed,
        offline=args.offline,
    )
