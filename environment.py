import random
import math

class Environment:
    """Mock OpenEnv base class for local testing without the openenv library installed."""
    def __init__(self, **kwargs):
        pass
    def reset(self):
        raise NotImplementedError
    def step(self, action):
        raise NotImplementedError
    def state(self):
        raise NotImplementedError

try:
    from openenv import Environment
except ImportError:
    pass

class PricingProEnv(Environment):
    """
    Pricing Pro Environment: Negotiation Loop with Half-Day Ticks
    """
    
    def __init__(self, use_llm_consumers=False, max_days=7, **kwargs):
        super().__init__(**kwargs)
        self.use_llm_consumers = use_llm_consumers
        self.max_days = max_days
        self.max_ticks = max_days * 2 # Morning and Afternoon shifts
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
        
    def reset(self):
        """Initializes inventory and the 9 Macroeconomic Parameters."""
        self.current_tick = 1
        self.retries = 0
        self.last_critique = ""
        self.history = []
        self.previous_prices = {}
        self.price_history = []
        self.rejection_count = 0
        
        # 1. Initialize Inventory
        self.inventory = {
            "Premium_Watch": {"stock": 50, "cost": 150.0},
            "Standard_Earbuds": {"stock": 200, "cost": 35.0},
            "Budget_Cable": {"stock": 500, "cost": 5.0}
        }
        
        # 2. Initialize 9 Macroeconomic Parameters
        self.macro_params = {
            "inflation_rate": round(random.uniform(0.01, 0.15), 2),
            "consumer_sentiment": round(random.uniform(0.1, 1.0), 2),
            "supply_chain_disruption": round(random.uniform(0.0, 1.0), 2),
            "competitor_aggressiveness": round(random.uniform(0.5, 1.5), 2),
            "seasonal_demand_multiplier": random.choice([0.8, 1.0, 1.5, 2.0]),
            "market_growth_rate": round(random.uniform(-0.05, 0.10), 2),
            "price_sensitivity_index": round(random.uniform(0.5, 2.0), 2),
            "brand_preference_bias": round(random.uniform(0.5, 2.0), 2),
            "purchase_urgency": round(random.uniform(0.5, 2.0), 2)
        }
        
        return self.state()
        
    def state(self):
        """Returns the observation dashboard for the AI-CFO."""
        day = ((self.current_tick - 1) // 2) + 1
        time_of_day = "Morning" if self.current_tick % 2 != 0 else "Afternoon"
        
        observation = f"--- DAY {day} OF {self.max_days} | Shift: {time_of_day} ---\n"
        
        if self.last_critique:
            observation += f"\n[!] NEGOTIATION COUNCIL REJECTED PREVIOUS ATTEMPT [!]\nCritique: {self.last_critique}\nAttempt {self.retries}/{self.max_retries}. Adjust prices and try again.\n\n"
        
        observation += (
            f"MACROECONOMIC CONDITIONS:\n"
            f"- Inflation Rate: {self.macro_params['inflation_rate'] * 100}%\n"
            f"- Consumer Sentiment: {self.macro_params['consumer_sentiment']} (0-1 scale)\n"
            f"- Supply Chain Disruption: {self.macro_params['supply_chain_disruption']} (0-1 scale)\n"
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
        if not isinstance(action, dict) or 'prices' not in action:
            self.last_critique = "Invalid JSON format. Must contain 'prices' key."
            self.retries += 1
            self.rejection_count += 1
            return self.state(), -10.0, False, {
                "error": "Invalid format",
                "council_rejections": self.rejection_count
            }
            
        ai_prices = action['prices']
        
        # --- COUNCIL NEGOTIATION REVIEW ---
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
            
            # Competitor check
            comp_base_margin = 1.20
            comp_markup = comp_base_margin + (self.macro_params['supply_chain_disruption'] * 0.5) 
            comp_markup = comp_markup / self.macro_params['competitor_aggressiveness']
            competitor_price = round(cost * comp_markup, 2)
            
            # Hard Rejection: Below Cost
            if margin < 0:
                reject_reasons.append(f"{product} is priced below cost (${price} vs cost ${cost}).")
            
            # Hard Rejection: > 3x Competitor
            elif price > competitor_price * 3:
                reject_reasons.append(f"{product} is priced 3x higher than competitor (${price} vs ${competitor_price}). Price gouging rejected.")
            
            # Soft Warning: > 1.5x Competitor but < 3x
            elif price > competitor_price * 1.5:
                soft_warnings.append(f"{product} is significantly above competitor (${price} vs ${competitor_price}), demand risk.")
                soft_penalty = True
                
        # If the Council finds issues, they reject and time DOES NOT advance
        if reject_reasons:
            self.retries += 1
            self.rejection_count += 1
            self.last_critique = " | ".join(reject_reasons)
            
            penalty = -10.0
            done = False
            
            # Max Retries Threshold
            if self.retries >= self.max_retries:
                # Force advance time to stop infinite loops
                self.current_tick += 1
                self.retries = 0
                self.last_critique = ""
                penalty = -20.0 # Final massive penalty for failing entirely
                
                all_sold_out = all(d['stock'] <= 0 for d in self.inventory.values())
                done = self.current_tick > self.max_ticks or all_sold_out
                
            info = {
                "status": "Council Rejected", 
                "critique": self.last_critique,
                "reward_breakdown": {
                    "profit": 0.0,
                    "inventory_penalty": 0.0,
                    "stability_penalty": 0.0,
                    "competitiveness_score": 0.0
                },
                "price_history": list(self.price_history),
                "council_rejections": self.rejection_count
            }
            return self.state(), penalty, done, info
            
        # --- APPROVAL & SALES EXECUTION ---
        self.last_critique = ""
        self.retries = 0
        
        daily_profit = 0.0
        stability_penalty = 0.0
        competitiveness_score = 0.0
        daily_sales_log = {}
        competitor_strategies = {}
        
        for product, details in self.inventory.items():
            if details['stock'] <= 0: continue
            
            ai_price = max(float(ai_prices[product]), 0.01)
            cost = details['cost']
            profit_margin = ai_price - cost
            
            comp_base_margin = 1.20
            comp_markup = comp_base_margin + (self.macro_params['supply_chain_disruption'] * 0.5) 
            comp_markup = comp_markup / self.macro_params['competitor_aggressiveness']
            competitor_price = round(cost * comp_markup, 2)
            
            base_price = competitor_price
            reaction_strength = 0.05
            strategy = "baseline"
            
            if product in self.previous_prices:
                our_previous_price = self.previous_prices[product]
                if our_previous_price > base_price * 1.1:
                    competitor_price = base_price * (1 - reaction_strength)
                    strategy = "undercutting"
                elif our_previous_price < base_price * 0.9:
                    competitor_price = base_price * (1 - reaction_strength * 0.5)
                    strategy = "defensive_match"
                else:
                    competitor_price = base_price * (1 + reaction_strength * 0.5)
                    strategy = "margin_hold"
            
            competitor_price = max(competitor_price, cost * 1.05)
            competitor_price = round(competitor_price, 2)
            competitor_strategies[product] = strategy
            
            # Pricing Stability Component
            if product in self.previous_prices:
                prev_price = self.previous_prices[product]
                if prev_price > 0:
                    pct_change = abs(ai_price - prev_price) / prev_price
                    if pct_change > 0.2: # >20% jump
                        stability_penalty -= (pct_change * 1.5)
                        
            # Competitiveness Component
            price_ratio = ai_price / competitor_price
            competitiveness_score -= (price_ratio - 1.0)
            
            # Half-day buyers
            base_buyers = 10 
            daily_buyers = int(base_buyers * self.macro_params['seasonal_demand_multiplier'] * self.macro_params['consumer_sentiment'])
            
            units_sold = 0
            
            for _ in range(daily_buyers):
                if units_sold >= self.inventory[product]['stock']: break
                    
                brand_score = self.macro_params['brand_preference_bias'] * 1.5
                purchase_urgency = max(self.macro_params['purchase_urgency'], 1e-6)
                price_penalty = self.macro_params['price_sensitivity_index'] / purchase_urgency
                
                our_value = brand_score / (ai_price ** price_penalty)
                comp_value = 1.0 / (competitor_price ** price_penalty)
                
                score_diff = our_value - comp_value
                k = 5.0
                x = max(min(k * score_diff, 10), -10)
                p_buy = 1.0 / (1.0 + math.exp(-x))
                
                if random.random() < p_buy:
                    units_sold += 1
                    
            self.inventory[product]['stock'] -= units_sold
            item_profit = units_sold * profit_margin
            daily_profit += item_profit
            
            daily_sales_log[product] = {
                "ai_price": ai_price,
                "competitor_price": competitor_price,
                "units_sold": units_sold,
                "profit_generated": round(item_profit, 2)
            }

        # Update History
        self.previous_prices = {k: float(v) for k, v in ai_prices.items() if k in self.inventory}
        self.price_history.append(self.previous_prices.copy())
        if len(self.price_history) > 5:
            self.price_history.pop(0)

        # Inventory Clearance Penalty
        inventory_penalty = 0.0
        if self.current_tick >= self.max_ticks * 0.8:
            for details in self.inventory.values():
                if details['stock'] > 0:
                    inventory_penalty -= details['stock'] * 0.05 # Penalty per unsold
                    
        self.current_tick += 1
        all_sold_out = all(d['stock'] <= 0 for d in self.inventory.values())
        done = self.current_tick > self.max_ticks or all_sold_out
        
        # End of Episode Penalty
        if done:
            for details in self.inventory.values():
                if details['stock'] > 0:
                    inventory_penalty -= (details['stock'] * details['cost'] * 0.05) # 5% cost penalty for unsold at end
                    
        if soft_penalty:
            competitiveness_score -= 1.0

        info = {
            "status": "Council Approved",
            "sales_log": daily_sales_log,
            "inventory_status": {k: v['stock'] for k, v in self.inventory.items()},
            "total_profit_this_tick": round(daily_profit, 2),
            "reward_breakdown": {
                "profit": round(daily_profit, 2),
                "inventory_penalty": round(inventory_penalty, 2),
                "stability_penalty": round(stability_penalty, 2),
                "competitiveness_score": round(competitiveness_score, 2)
            },
            "price_history": list(self.price_history),
            "council_rejections": self.rejection_count,
            "soft_warnings": soft_warnings,
            "competitor_strategy": competitor_strategies,
            "competitor_strategy_summary": list(set(competitor_strategies.values()))
        }
        
        self.history.append(info)
        reward = round(daily_profit + inventory_penalty + stability_penalty + competitiveness_score, 2)
        reward = max(min(reward, 100.0), -100.0)
        
        return self.state(), reward, done, info
