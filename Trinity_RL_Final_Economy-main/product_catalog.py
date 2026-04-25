import random


# Broad catalog to avoid overfitting pricing behavior to one niche.
PRODUCT_CATALOG = [
    {"name": "Premium_Watch", "cost_range": (120.0, 220.0)},
    {"name": "Standard_Earbuds", "cost_range": (20.0, 60.0)},
    {"name": "Budget_Cable", "cost_range": (2.0, 12.0)},
    {"name": "Gaming_Mouse", "cost_range": (18.0, 55.0)},
    {"name": "Mechanical_Keyboard", "cost_range": (35.0, 120.0)},
    {"name": "Smart_Speaker", "cost_range": (25.0, 90.0)},
    {"name": "Designer_Handbag", "cost_range": (180.0, 420.0)},
    {"name": "Leather_Boots", "cost_range": (45.0, 130.0)},
    {"name": "Cotton_Tee", "cost_range": (6.0, 22.0)},
    {"name": "Running_Shoes", "cost_range": (28.0, 95.0)},
    {"name": "Blender", "cost_range": (20.0, 85.0)},
    {"name": "Air_Fryer", "cost_range": (35.0, 140.0)},
    {"name": "Coffee_Beans_1kg", "cost_range": (8.0, 30.0)},
    {"name": "Protein_Powder", "cost_range": (16.0, 55.0)},
    {"name": "Office_Chair", "cost_range": (45.0, 210.0)},
]


def sample_products(min_items=3, max_items=5):
    count = random.randint(min_items, max_items)
    choices = random.sample(PRODUCT_CATALOG, k=min(count, len(PRODUCT_CATALOG)))
    return [
        {
            "name": item["name"],
            "cost": round(random.uniform(item["cost_range"][0], item["cost_range"][1]), 2),
        }
        for item in choices
    ]
