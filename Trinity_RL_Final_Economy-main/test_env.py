from environment import PricingProEnv
import json

def test():
    print("Initializing Negotiation-Enabled Pricing Pro Environment...")
    env = PricingProEnv()
    
    state = env.reset()
    
    done = False
    total_reward = 0
    
    # Let's test the Council Rejection by purposefully pricing below cost
    print("\n" + "="*50)
    print(f"=== OBSERVATION ===")
    print(state)
    
    bad_action = {
        "prices": {
            "Premium_Watch": 149.0, # Cost is 150, so this is a slight loss!
            "Standard_Earbuds": 49.99,
            "Budget_Cable": 9.99
        }
    }
    
    print("\n=== AI-CFO ACTION (INTENTIONAL SLIGHT LOSS) ===")
    print(json.dumps(bad_action, indent=4))
    
    next_state, reward, done, info = env.step(bad_action)
    total_reward += reward
    
    print("\n=== STEP RESULT ===")
    print(f"Penalty (Reward): {reward}")
    print("Info:")
    print(json.dumps(info, indent=4))
    
    # Now, let's correct it on the second try (Retry 1)
    state = next_state
    print("\n" + "="*50)
    print(f"=== OBSERVATION AFTER REJECTION ===")
    print(state)
    
    good_action = {
        "prices": {
            "Premium_Watch": 199.99, # Corrected price!
            "Standard_Earbuds": 49.99,
            "Budget_Cable": 9.99
        }
    }
    
    print("\n=== AI-CFO ACTION (CORRECTED) ===")
    print(json.dumps(good_action, indent=4))
    
    next_state, reward, done, info = env.step(good_action)
    total_reward += reward
    
    print("\n=== STEP RESULT ===")
    print(f"Profit (Reward): {reward}")
    print("Info:")
    print(json.dumps(info, indent=4))

if __name__ == "__main__":
    test()
