import math
import random
from council import Council

class ReviewerEngine:
    def __init__(self, council_instance=None):
        self.council = council_instance or Council()
        
    def heuristic_review(self, ai_prices, dna_dict, products, macro_params, competitor_prices):
        """
        FAST logic using the EXACT probabilistic physics from the original environment.
        """
        rejections = []
        items_bought = 0
        total_reward = 0.0
        
        # DNA Modulation (From your original logic)
        dna_sense = 1.5 if dna_dict['intelligence'] == "Researcher" else 1.0
        dna_urgency = 2.0 if dna_dict['urgency'] == "Panic-Buyer" else 1.0

        for product_name, price in ai_prices.items():
            product = next((p for p in products if p['name'] == product_name), None)
            if not product:
                continue
                
            cost = product['cost']
            comp_price = competitor_prices.get(product_name, cost * 1.2)
            
            try:
                price = max(float(price), 0.01)
            except (ValueError, TypeError):
                price = 0.01
            comp_price = max(float(comp_price), 0.01)
            
            # 1. Hard Check: Below Cost
            if price < cost:
                rejections.append(f"{product_name} below cost")
                total_reward -= 20.0
                continue

            # 2. Market Physics (Your Original Formula)
            brand_score = macro_params.get('brand_preference_bias', 1.0) * 1.5
            urgency = max(macro_params.get('purchase_urgency', 1.0) * dna_urgency, 1e-6)
            price_penalty = (macro_params.get('price_sensitivity_index', 1.0) * dna_sense) / urgency
            
            # Add epsilon to base to prevent ZeroDivisionError/MathDomainError
            our_value = brand_score / ((price + 1e-6) ** price_penalty)
            comp_value = 1.0 / ((comp_price + 1e-6) ** price_penalty)
            
            score_diff = our_value - comp_value
            k = 5.0
            x = max(min(k * score_diff, 10), -10)
            p_buy = 1.0 / (1.0 + math.exp(-x))
            
            if random.random() < p_buy:
                items_bought += 1
                margin = (price - cost) / cost
                # Cap the reward to prevent reward hacking through astronomical prices
                total_reward += max(min(margin * 10.0, 50.0), -20.0)
            else:
                rejections.append(f"Price for {product_name} rejected by {dna_dict['economic']} budget.")
                total_reward -= 2.0

        if not ai_prices:
            success = False
        else:
            success = items_bought / len(ai_prices) >= 0.5
            
        critique = " | ".join(rejections) if not success else "SUCCESS"
        
        return success, total_reward, critique

    def council_review(self, ai_prices, scenario_text, dna_dict, products):
        """Used in Phase 3 (Council Polish) and Demo."""
        prompt = (
            f"Character: {scenario_text}\n"
            f"Prices: {ai_prices}\n"
            "Task: Evaluate the pricing. Output ONLY a valid JSON object in the exact format: "
            '{"status": "SUCCESS", "score": 10.0} or {"status": "FAIL", "score": -10.0}. '
            "Do not include any other text."
        )
        response = self.council._call_groq(self.council.narrator_model, prompt)
        
        try:
            import json
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                success = data.get("status") == "SUCCESS"
                score = float(data.get("score", 10.0 if success else -10.0))
                # Clip the score to prevent wild reward values
                score = max(min(score, 50.0), -50.0)
                return success, score, response
        except Exception:
            pass
            
        # Fallback with stricter parsing
        success = "SUCCESS" in response.upper() and "FAIL" not in response.upper()
        return success, 10.0 if success else -10.0, response
