import json
import random
from persona_engine import PersonaEngine
from council import Council

def generate_sft_dataset(num_samples=200):
    engine = PersonaEngine()
    council = Council()
    dataset = []

    print(f"Generating {num_samples} SFT samples...")

    for i in range(num_samples):
        dna = engine.generate_random_dna()
        dna_dict = engine.dna_to_dict(dna)
        
        # Randomize Macro Params for variety
        macro_params = {
            "inflation": round(random.uniform(0.01, 0.15), 2),
            "sentiment": round(random.uniform(0.1, 1.0), 2)
        }
        products = council.generate_products(dna_dict, macro_params, min_items=3, max_items=5)

        scenario = council.generate_scenario(dna_dict, products, macro_params)
        golden_prices = council.generate_golden_price(dna_dict, products, macro_params)

        sample = {
            "instruction": f"Set prices for the following products based on the market scenario.\n\nScenario: {scenario}",
            "input": f"Products: {json.dumps(products)}\nMarket Conditions: {json.dumps(macro_params)}",
            "output": json.dumps({"reasoning": "Determined by council logic.", "prices": golden_prices}, indent=2)
        }
        dataset.append(sample)

    with open("sft_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Successfully saved {num_samples} samples to sft_dataset.json")

if __name__ == "__main__":
    generate_sft_dataset(200)
