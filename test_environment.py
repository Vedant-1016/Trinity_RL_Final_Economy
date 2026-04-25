import math
import random
from environment import PricingProEnv

def test_basic_sanity():
    print("\n" + "="*50)
    print("TEST 1: Basic Sanity Test")
    print("="*50)
    env = PricingProEnv()
    env.reset()
    
    for i in range(10):
        action = {
            "prices": {
                "Premium_Watch": 200.0,
                "Standard_Earbuds": 45.0,
                "Budget_Cable": 7.0
            }
        }
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Reward = {reward:7.2f} | Status = {info.get('status', 'Unknown'):<18} | Profit = {info.get('total_profit_this_tick', 0):7.2f} | Inventory = {info.get('inventory_status')}")
        if done:
            print("Episode done.")
            break

def test_council_rejection():
    print("\n" + "="*50)
    print("TEST 2: Council Rejection Test")
    print("="*50)
    env = PricingProEnv()
    env.reset()
    
    # Missing product & Below cost
    action = {
        "prices": {
            "Premium_Watch": 100.0,  # Below cost of 150
            "Standard_Earbuds": 60.0 # High, valid
            # Missing Budget_Cable
        }
    }
    
    obs, reward, done, info = env.step(action)
    print(f"Status:   {info.get('status')}")
    print(f"Critique: {info.get('critique')}")
    print(f"Penalty:  {reward:.2f}")

def test_retry_mechanism():
    print("\n" + "="*50)
    print("TEST 3: Retry Mechanism Test")
    print("="*50)
    env = PricingProEnv()
    env.reset()
    
    max_retries = env.max_retries
    for i in range(max_retries + 1):
        action = {
            "prices": {
                "Premium_Watch": 10.0, # Consistently invalid
                "Standard_Earbuds": 10.0,
                "Budget_Cable": 1.0
            }
        }
        obs, reward, done, info = env.step(action)
        print(f"Attempt {i+1}: Status = {info.get('status'):<18} | Reward = {reward:6.2f} | Retries Count = {env.retries}")

def test_stability():
    print("\n" + "="*50)
    print("TEST 4: Stability Test")
    print("="*50)
    env = PricingProEnv()
    env.reset()
    
    rewards = []
    steps = 0
    
    for _ in range(50):
        # Ensure we always provide valid prices to avoid council loops
        action = {
            "prices": {
                "Premium_Watch": random.uniform(160, 300),
                "Standard_Earbuds": random.uniform(40, 80),
                "Budget_Cable": random.uniform(6, 15)
            }
        }
        obs, reward, done, info = env.step(action)
        if "Approved" in info.get("status", ""):
            rewards.append(reward)
            steps += 1
        
        if done:
            break
            
    print(f"Executed steps: {steps}")
    if any(math.isnan(r) for r in rewards):
        print("FAIL: NaN reward detected.")
    else:
        print("PASS: No NaN rewards.")
        
    if rewards:
        print(f"Reward Range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    else:
        print("No valid rewards collected.")

def test_multi_agent():
    print("\n" + "="*50)
    print("TEST 5: Multi-Agent Behavior Test")
    print("="*50)
    
    # Case A: High Pricing
    print("--- Case A: High Pricing ---")
    env = PricingProEnv()
    env.reset()
    
    for i in range(3):
        action = {
            "prices": {
                "Premium_Watch": 300.0, # High, but won't trigger >3x hard rejection
                "Standard_Earbuds": 70.0,
                "Budget_Cable": 10.0
            }
        }
        obs, reward, done, info = env.step(action)
        if 'sales_log' in info and 'Premium_Watch' in info['sales_log']:
            sold = info['sales_log']['Premium_Watch']['units_sold']
            profit = info['sales_log']['Premium_Watch']['profit_generated']
            strategy = info.get('competitor_strategy', {}).get('Premium_Watch', 'N/A')
            print(f"Step {i+1} | Strategy: {strategy:<16} | Units Sold: {sold:<2} | Profit: {profit}")

    # Case B: Competitive Pricing
    print("\n--- Case B: Competitive Pricing ---")
    env = PricingProEnv()
    env.reset()
    
    for i in range(3):
        action = {
            "prices": {
                "Premium_Watch": 180.0, # Reasonable markup
                "Standard_Earbuds": 45.0,
                "Budget_Cable": 7.0
            }
        }
        obs, reward, done, info = env.step(action)
        if 'sales_log' in info and 'Premium_Watch' in info['sales_log']:
            sold = info['sales_log']['Premium_Watch']['units_sold']
            profit = info['sales_log']['Premium_Watch']['profit_generated']
            strategy = info.get('competitor_strategy', {}).get('Premium_Watch', 'N/A')
            print(f"Step {i+1} | Strategy: {strategy:<16} | Units Sold: {sold:<2} | Profit: {profit}")

def test_logging_validation():
    print("\n" + "="*50)
    print("TEST 6: Logging Validation")
    print("="*50)
    env = PricingProEnv()
    env.reset()
    
    action = {
        "prices": {
            "Premium_Watch": 200.0,
            "Standard_Earbuds": 50.0,
            "Budget_Cable": 8.0
        }
    }
    obs, reward, done, info = env.step(action)
    
    expected_keys = ['reward_breakdown', 'competitor_strategy', 'price_history', 'council_rejections']
    all_passed = True
    for key in expected_keys:
        if key in info:
            print(f"PASS: Found '{key}' in info.")
        else:
            print(f"FAIL: Missing '{key}' in info.")
            all_passed = False
            
    if all_passed:
        print("\nReward Breakdown:")
        for k, v in info.get('reward_breakdown', {}).items():
            print(f"  {k}: {v}")
            
        print(f"\nCompetitor Strategy Summary: {info.get('competitor_strategy_summary')}")

if __name__ == '__main__':
    test_basic_sanity()
    test_council_rejection()
    test_retry_mechanism()
    test_stability()
    test_multi_agent()
    test_logging_validation()
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED SUCCESSFULLY.")
    print("="*50)
