import os
import json
import re
from groq import Groq
from dotenv import load_dotenv
from product_catalog import sample_products

load_dotenv()

class Council:
    def __init__(self, api_key=None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        # Allow running without external API credentials (e.g., local smoke tests,
        # CI, or Hugging Face Spaces builds). In that case we fall back to simple,
        # deterministic text generation and heuristic pricing.
        self.client = Groq(api_key=api_key) if api_key else None
        # Using 3 distinct models available on Groq
        self.narrator_model = "llama-3.1-70b-versatile"
        self.analyst_model = "mixtral-8x7b-32768"
        self.synthesizer_model = "gemma2-9b-it"
        self.product_model = "llama-3.1-8b-instant"

    def _extract_json(self, text):
        if not isinstance(text, str):
            return None
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            return None
        return None

    def _validate_products(self, products, min_items=3, max_items=5):
        if not isinstance(products, list):
            return None

        cleaned = []
        seen = set()
        for item in products:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip().replace(" ", "_")
            if not name or name in seen:
                continue
            try:
                cost = float(item.get("cost", 0.0))
            except Exception:
                continue

            # Clamp to reasonable "everyday product" training range.
            cost = round(max(1.0, min(cost, 500.0)), 2)
            cleaned.append({"name": name, "cost": cost})
            seen.add(name)

        if len(cleaned) < min_items:
            return None
        return cleaned[:max_items]

    def generate_products(self, dna_dict, macro_params, min_items=3, max_items=5):
        prompt = (
            "You are the Product Curator for a retail pricing simulation.\n"
            "Task: Propose a small basket of everyday, simple consumer products with moderate variety.\n"
            f"Use this customer DNA: {dna_dict}\n"
            f"Use these macro conditions: {macro_params}\n"
            f"Return ONLY valid JSON with this exact schema: {{\"products\": [{{\"name\": \"string_with_underscores\", \"cost\": 12.34}}]}}\n"
            f"Rules:\n"
            f"- Generate between {min_items} and {max_items} products\n"
            "- Keep products common and practical (avoid niche/industrial items)\n"
            "- Include light variety in price points (budget/mid/premium)\n"
            "- cost must be positive and realistic\n"
            "- no extra text"
        )
        response = self._call_groq(self.product_model, prompt)
        data = self._extract_json(response)
        if data and "products" in data:
            validated = self._validate_products(data["products"], min_items=min_items, max_items=max_items)
            if validated:
                return validated
        return sample_products(min_items=min_items, max_items=max_items)

    def generate_scenario(self, dna_dict, products, macro_params):
        """
        Collaborative interaction loop between 3 models for a rich output.
        """
        # Step 1: Narrator creates the base story
        base_story = self._call_groq(
            self.narrator_model,
            f"Create a 2-sentence market scenario for a customer with this DNA: {dna_dict}. Products: {products}."
        )

        # Step 2: Analyst adds psychological depth
        critique = self._call_groq(
            self.analyst_model,
            f"Review this scenario: '{base_story}'. Add a specific psychological trait or quirk consistent with the DNA: {dna_dict}."
        )

        # Step 3: Synthesizer creates the final 'Rich' narrative
        final_scenario = self._call_groq(
            self.synthesizer_model,
            f"Merge these into one polished, engaging 3-sentence narrative for an AI-CFO to read. Story: {base_story}. Trait: {critique}. Do not use labels, just storytelling."
        )

        final_scenario = (final_scenario or "").strip()
        if final_scenario and final_scenario != "CALL_FAILED":
            return final_scenario

        # Offline fallback (no API key / call failure).
        inflation = macro_params.get("inflation")
        sentiment = macro_params.get("sentiment")
        product_names = [p.get("name") for p in (products or []) if isinstance(p, dict) and p.get("name")]
        basket = ", ".join(product_names[:5]) if product_names else "a small basket of everyday goods"
        return (
            f"Market conditions show inflation={inflation} and sentiment={sentiment}. "
            f"The customer is evaluating {basket} and is price-sensitive but still values perceived quality. "
            "Competitors are adjusting prices cautiously, creating opportunities for smart, defensible margins."
        )

    def generate_golden_price(self, dna_dict, products, macro_params):
        """
        Council collaborates to decide the 'Optimal' price for SFT.
        """
        analysis = self._call_groq(
            self.narrator_model,
            f"As a master economist, given DNA {dna_dict} and Macro {macro_params}, what is the optimal profit margin percentage for {products}? Output ONLY the percentage number."
        )
        
        try:
            margin = float(analysis.strip().replace('%', '')) / 100
        except:
            margin = 0.25 # Fallback
            
        return {p['name']: round(p['cost'] * (1 + margin), 2) for p in products}

    def _call_groq(self, model, prompt):
        if self.client is None:
            return "CALL_FAILED"
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return completion.choices[0].message.content
        except Exception:
            # Avoid leaking detailed backend/runtime errors into model-facing text.
            return "CALL_FAILED"
