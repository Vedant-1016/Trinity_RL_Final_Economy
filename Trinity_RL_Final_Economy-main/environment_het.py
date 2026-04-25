import random
import math
import json
from persona_engine import PersonaEngine
from reviewer_engine import ReviewerEngine

class Environment:
    """Mock OpenEnv base class for local testing."""
    def __init__(self, **kwargs): pass
    def reset(self): raise NotImplementedError
    def step(self, action): raise NotImplementedError
    def state(self): raise NotImplementedError

try:
    from openenv import Environment
except ImportError:
    pass

class PricingProEnv(Environment):
    """
    Pricing Pro Environment: Restored original logic with Persona DNA Extensions.
    NO hardcoded products or values.
    """
    
    def __init__(self, products=None, macro_config=None, use_llm_consumers=False, max_days=7, **kwargs):
        super().__init__(**kwargs)
        self.use_llm_consumers = use_llm_consumers
        self.max_days = max_days
        self.max_ticks = max_days * 2 
        
        # 1. Non-hardcoded configuration
        self.products_list = products or [
            {"name": "Premium_Watch", "cost": 150.0, "stock": 50},
            {"name": "Standard_Earbuds", "cost": 35.0, "stock": 200},
            {"name": "Budget_Cable", "cost": 5.0, "stock": 500}
        ]
        self.inventory_config = {p['name']: {"stock": p['stock'], "cost": p['cost']} for p in self.products_list}
        self.macro_config = macro_config or {
            "inflation_rate": (0.01, 0.15),
            "consumer_sentiment": (0.1, 1.0),
            "supply_chain_disruption": (0.0, 1.0),
            "competitor_aggressiveness": (0.5, 1.5),
            "seasonal_multipliers": [0.8, 1.0, 1.5, 2.0],
            "market_growth": (-0.05, 0.10),
            "price_sensitivity": (0.5, 2.0),
            "brand_preference": (0.5, 2.0),
            "purchase_urgency": (0.5, 2.0)
        }

        self.persona_engine = PersonaEngine()
        self.reviewer_engine = ReviewerEngine()
        
        self.current_tick = 0
        self.retries = 0
        self.max_retries = 3
        self.last_critique = ""
        self.inventory = {}
        self.macro_params = {}
        self.history = []
        self.previous_prices = {}
        self.price_history = []
        self.rejection_count = 0
        self.current_persona_dna = None

    def reset(self, demo_context=None):
        """Initializes a new Sell Day scenario with optional demo context."""
        self.current_tick = 1
        self.total_profit = 0.0
        self.demo_context = demo_context
        
        # Set macro params based on demo context if provided
        self.macro_params = {
            "inflation_rate": round(random.uniform(0.01, 0.15), 2),
            "consumer_sentiment": 0.8 if demo_context and "Spending" in demo_context.get('vibe', '') else 0.4,
            "competitor_aggressiveness": 1.5 if demo_context and "Price Cutter" in demo_context.get('rival', '') else 1.0,
            "purchase_urgency": 1.8 if demo_context and "Rush" in demo_context.get('day_type', '') else 1.0,
            "brand_preference_bias": 1.5,
            "price_sensitivity_index": 1.2,
            "seasonal_demand_multiplier": 1.0,
            "supply_chain_disruption": 0.2,
            "market_growth_rate": 0.02
        }
        
        return self.generate_new_scenario()

    def generate_new_scenario(self):
        """Generates a fresh Persona and Scenario using the Council for the demo."""
        dna_tuple = self.persona_engine.generate_random_dna()
        self.current_persona_dna = self.persona_engine.dna_to_dict(dna_tuple)
        
        # In demo mode (if use_llm_consumers is True), we ask the Council to write the story
        if self.use_llm_consumers:
            scenario_text = self.reviewer_engine.council.generate_scenario(
                self.current_persona_dna, 
                self.products_list, 
                self.macro_params
            )
        else:
            scenario_text = f"A customer with {self.current_persona_dna['economic']} budget is looking for items."

        day = (self.current_tick + 1) // 2
        shift = "Morning" if self.current_tick % 2 != 0 else "Afternoon"
        
        return {
            "day": day,
            "shift": shift,
            "scenario": scenario_text,
            "dna": self.current_persona_dna,
            "inventory": {k: v['stock'] for k, v in self.inventory.items()},
            "instruction": "The Council has presented a new customer. Set your prices."
        }

    def step(self, action):
        """Processes the AI-CFO's pricing using either Heuristic or Council Buyer."""
        ai_prices = action['prices']
        
        # A. Negotiation Council (The 'Hard' Rejection Logic)
        # (This part is preserved from original for demo validation)
        
        # B. Buyer Decision (Council or Heuristic)
        if self.use_llm_consumers:
            # The Council acts as the customer and decides whether to buy
            obs = self.state() # Get current narrative
            success, reward, critique = self.reviewer_engine.council_review(
                ai_prices, obs, self.current_persona_dna, self.products_list
            )
        else:
            # Default to fast heuristic for training
            competitor_prices = {p: round(v['cost'] * 1.2, 2) for p, v in self.inventory.items()}
            success, reward, critique = self.reviewer_engine.heuristic_review(
                ai_prices, self.current_persona_dna, self.products_list, self.macro_params, competitor_prices
            )

        # C. Inventory and Profit Update
        shift_profit = 0.0
        if success:
            for product, price in ai_prices.items():
                if self.inventory[product]['stock'] > 0:
                    self.inventory[product]['stock'] -= 1
                    shift_profit += (price - self.inventory[product]['cost'])
            self.total_profit += shift_profit

    def state(self):
        """Returns the observation dashboard for the AI-CFO."""
        day = ((self.current_tick - 1) // 2) + 1
        time_of_day = "Morning" if self.current_tick % 2 != 0 else "Afternoon"
        
        observation = f"--- DAY {day} OF {self.max_days} | Shift: {time_of_day} ---\n"
        
        if self.last_critique:
            observation += f"\n[!] NEGOTIATION COUNCIL REJECTED PREVIOUS ATTEMPT [!]\nCritique: {self.last_critique}\nAttempt {self.retries}/{self.max_retries}. Adjust prices and try again.\n\n"
        
        # Adding Persona DNA to the observation
        observation += f"CURRENT CUSTOMER PROFILE (DNA):\n{json.dumps(self.current_persona_dna, indent=2)}\n\n"
        
        observation += (
            f"MACROECONOMIC CONDITIONS:\n"
            f"- Inflation Rate: {self.macro_params['inflation_rate'] * 100}%\n"
            f"- Consumer Sentiment: {self.macro_params['consumer_sentiment']}\n"
            f"- Supply Chain Disruption: {self.macro_params['supply_chain_disruption']}\n"
            f"- Competitor Aggressiveness: {self.macro_params['competitor_aggressiveness']}\n"
            f"- Seasonal Demand Multiplier: x{self.macro_params['seasonal_demand_multiplier']}\n"
            f"- Market Growth Rate: {self.macro_params['market_growth_rate'] * 100}%\n"
            f"- Price Sensitivity Index: {self.macro_params['price_sensitivity_index']}\n"
            f"- Brand Preference Bias: {self.macro_params['brand_preference_bias']}\n"
            f"- Purchase Urgency: {self.macro_params['purchase_urgency']}\n\n"
            f"INVENTORY & COSTS:\n"
        )
        
        for item, details in self.inventory.items():
            observation += f"- {item}: {details['stock']} units left (Cost: ${details['cost']})\n"
            
        observation += "\nTASK: Output a JSON dictionary with key 'prices' containing a map of Product Name to your chosen float price."
        return observation

    def step(self, action):
        """Exactly the original step logic, with DNA influence injected."""
        if not isinstance(action, dict) or 'prices' not in action:
            self.last_critique = "Invalid JSON format. Must contain 'prices' key."
            self.retries += 1
            self.rejection_count += 1
            return self.state(), -10.0, False, {"error": "Invalid format", "council_rejections": self.rejection_count}
            
        ai_prices = action['prices']
        
        # --- COUNCIL NEGOTIATION REVIEW (RESTORED) ---
        reject_reasons = []
        soft_warnings = []
        soft_penalty = False
        
        for product, details in self.inventory.items():
            if details['stock'] <= 0: continue
            if product not in ai_prices:
                reject_reasons.append(f"Forgot to price {product}.")
                continue
                
            price = max(float(ai_prices[product]), 0.01)
            cost = details['cost']
            margin = (price - cost) / cost
            
            comp_base_margin = 1.20
            comp_markup = comp_base_margin + (self.macro_params['supply_chain_disruption'] * 0.5) 
            comp_markup = comp_markup / self.macro_params['competitor_aggressiveness']
            competitor_price = round(cost * comp_markup, 2)
            
            if margin < 0:
                reject_reasons.append(f"{product} is priced below cost (${price} vs cost ${cost}).")
            elif price > competitor_price * 3:
                reject_reasons.append(f"{product} is priced 3x higher than competitor (${price} vs ${competitor_price}). Price gouging rejected.")
            elif price > competitor_price * 1.5:
                soft_warnings.append(f"{product} is significantly above competitor (${price} vs ${competitor_price}), demand risk.")
                soft_penalty = True
                
        if reject_reasons:
            self.retries += 1
            self.rejection_count += 1
            self.last_critique = " | ".join(reject_reasons)
            penalty = -10.0
            done = False
            
            if self.retries >= self.max_retries:
                self.current_tick += 1
                self.retries = 0
                self.last_critique = ""
                penalty = -20.0
                all_sold_out = all(d['stock'] <= 0 for d in self.inventory.values())
                done = self.current_tick > self.max_ticks or all_sold_out
                # New DNA for the forced next tick
                dna_tuple = self.persona_engine.generate_random_dna()
                self.current_persona_dna = self.persona_engine.dna_to_dict(dna_tuple)
                
            return self.state(), penalty, done, {"status": "Council Rejected", "critique": self.last_critique, "council_rejections": self.rejection_count}
            
        # --- APPROVAL & SALES EXECUTION (RESTORED) ---
        self.last_critique = ""
        self.retries = 0
        
        daily_profit = 0.0
        stability_penalty = 0.0
        competitiveness_score = 0.0
        daily_sales_log = {}
        
        # DNA-Influence on the original math:
        dna_sense_mod = 1.5 if self.current_persona_dna['intelligence'] == "Researcher" else 1.0
        dna_urgency_mod = 2.0 if self.current_persona_dna['urgency'] == "Panic-Buyer" else 1.0

        for product, details in self.inventory.items():
            if details['stock'] <= 0: continue
            
            ai_price = max(float(ai_prices[product]), 0.01)
            cost = details['cost']
            profit_margin = ai_price - cost
            
            comp_base_margin = 1.20
            comp_markup = comp_base_margin + (self.macro_params['supply_chain_disruption'] * 0.5) 
            comp_markup = comp_markup / self.macro_params['competitor_aggressiveness']
            competitor_price = round(cost * comp_markup, 2)
            
            # Original Pricing Stability Component
            if product in self.previous_prices:
                prev_price = self.previous_prices[product]
                if prev_price > 0:
                    pct_change = abs(ai_price - prev_price) / prev_price
                    if pct_change > 0.2:
                        stability_penalty -= (pct_change * 1.5)
                        
            competitiveness_score -= (ai_price / competitor_price - 1.0)
            
            # Original Probabilistic Sales Formula
            base_buyers = 10 
            daily_buyers = int(base_buyers * self.macro_params['seasonal_demand_multiplier'] * self.macro_params['consumer_sentiment'])
            units_sold = 0
            
            for _ in range(daily_buyers):
                if units_sold >= self.inventory[product]['stock']: break
                    
                brand_score = self.macro_params['brand_preference_bias'] * 1.5
                purchase_urgency = max(self.macro_params['purchase_urgency'] * dna_urgency_mod, 1e-6)
                price_penalty = (self.macro_params['price_sensitivity_index'] * dna_sense_mod) / purchase_urgency
                
                # Add epsilon to base to prevent ZeroDivisionError/MathDomainError
                our_value = brand_score / ((ai_price + 1e-6) ** price_penalty)
                comp_value = 1.0 / ((competitor_price + 1e-6) ** price_penalty)
                
                score_diff = our_value - comp_value
                k = 5.0
                x = max(min(k * score_diff, 10), -10)
                p_buy = 1.0 / (1.0 + math.exp(-x))
                
                if random.random() < p_buy:
                    units_sold += 1
                    
            self.inventory[product]['stock'] -= units_sold
            item_profit = units_sold * profit_margin
            daily_profit += item_profit
            
            daily_sales_log[product] = {"ai_price": ai_price, "competitor_price": competitor_price, "units_sold": units_sold, "profit": round(item_profit, 2)}

        # Update History & Next Persona
        self.previous_prices = {k: float(v) for k, v in ai_prices.items() if k in self.inventory}
        self.price_history.append(self.previous_prices.copy())
        
        inventory_penalty = 0.0
        self.current_tick += 1
        
        # New DNA for the next customer
        dna_tuple = self.persona_engine.generate_random_dna()
        self.current_persona_dna = self.persona_engine.dna_to_dict(dna_tuple)
        
        all_sold_out = all(d['stock'] <= 0 for d in self.inventory.values())
        done = self.current_tick > self.max_ticks or all_sold_out
        
        reward = round(daily_profit + inventory_penalty + stability_penalty + competitiveness_score, 2)
        reward = max(min(reward, 100.0), -100.0)
        
        info = {
            "status": "Council Approved",
            "sales_log": daily_sales_log,
            "inventory_status": {k: v['stock'] for k, v in self.inventory.items()},
            "reward_breakdown": {"profit": round(daily_profit, 2), "stability": round(stability_penalty, 2), "comp_score": round(competitiveness_score, 2)},
            "dna_at_step": self.current_persona_dna
        }
        
        return self.state(), reward, done, info
