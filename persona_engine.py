import random
import itertools

class PersonaEngine:
    def __init__(self):
        # Dimension 1: Economic Profile (The "Wallet")
        self.economic_profiles = ["Fixed/Low", "Variable", "Stable", "High"]
        
        # Dimension 2: Primary Psychological Driver (The "Why")
        self.drivers = ["Frugality", "Performance", "Status", "Time-Sensitivity"]
        
        # Dimension 3: Market Intelligence (The "Awareness")
        self.intelligence = ["Researcher", "Loyalist", "Skeptic", "Impressionable"]
        
        # Dimension 4: Purchase Urgency (The "When")
        self.urgencies = ["Predator", "Just-in-Time", "Panic-Buyer"]
        
        # Dimension 5: Volume & Context (The "Scale")
        self.scales = ["Individual", "Bulk", "Gift-Giver"]

    def get_all_combinations(self):
        """Returns all 576 unique DNA combinations."""
        return list(itertools.product(
            self.economic_profiles, 
            self.drivers, 
            self.intelligence, 
            self.urgencies, 
            self.scales
        ))

    def generate_random_dna(self):
        """Returns a single random DNA tuple."""
        return (
            random.choice(self.economic_profiles),
            random.choice(self.drivers),
            random.choice(self.intelligence),
            random.choice(self.urgencies),
            random.choice(self.scales)
        )

    def dna_to_dict(self, dna):
        """Converts DNA tuple to a readable dictionary."""
        return {
            "economic": dna[0],
            "driver": dna[1],
            "intelligence": dna[2],
            "urgency": dna[3],
            "scale": dna[4]
        }
