# Pricing Pro: AI-CFO Market Simulation (OpenEnv Hackathon)

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue.svg)](openenv.yaml)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow.svg)](https://huggingface.co/spaces/your-username/pricing-pro-ai-cfo)
[![Training Notebook](https://img.shields.io/badge/Colab-Training%20Notebook-orange.svg)](notebooks/train_colab.ipynb)

**An OpenEnv reinforcement learning environment designed to teach LLMs emergent strategic behavior, dynamic pricing, and theory-of-mind through multi-agent market competition.**

---

## 🎯 The Problem

Teaching an LLM to simply output a number is easy. Teaching an LLM to understand *market dynamics*, anticipate a competitor's moves, and balance profit margins against brand loyalty in a partially observable environment is extremely difficult. 

Current training environments often rely on static datasets or simplistic 0/1 win/loss games. **Pricing Pro** tackles the "Multi-Agent Interactions" theme by introducing a dynamic, multi-actor market simulation. It forces the agent to model the beliefs and incentives of both competitors and varied consumer personas to discover the optimal pricing strategy.

---

## 🌍 The Environment

The Pricing Pro environment is built natively on the **OpenEnv** framework, providing a standard Gym-style API (`reset`, `step`, `state`) for seamless integration with Hugging Face TRL and Unsloth.

### How it Works (The Simulation Loop)
1. **The State (`state`):** The Environment generates a daily market context (e.g., "High inflation, consumers are price-sensitive") and baseline costs.
2. **The Action (`step`):** The LLM (acting as the AI-CFO) observes the state and outputs a price for its product.
3. **The Multi-Agent Interaction:**
    *   **The Competitor Agent:** Sets a competing price based on market noise.
    *   **The Consumer Council:** 3 distinct LLM agents (representing different buyer personas like "Bargain Hunter" or "Brand Loyalist") independently evaluate both prices and brand reputation, using theory-of-mind to decide which product offers better value.
    *   **The Chairman (Scalable Oversight):** Aggregates the consumer decisions and declares the final sales figures.

### The Reward Signal
We designed a mathematically rigorous reward signal that genuinely teaches strategic behavior rather than simple exploitation:

`Reward = (AI_CFO_Price - Cost_of_Goods) * Units_Sold`

*   **Priced too high?** The Competitor wins the consumers. 0 units sold = `0 Reward`.
*   **Priced too low?** The AI-CFO wins the consumers but takes a loss. 3 units sold at a loss = `Negative Reward`.
*   The LLM must explore and converge on the optimal profit-maximizing price point dynamically.

---

## Submission Deliverables

Replace the two external URLs below with your final public links before submission:

- Hugging Face Space (public): [https://huggingface.co/spaces/your-username/pricing-pro-ai-cfo](https://huggingface.co/spaces/your-username/pricing-pro-ai-cfo)
- Writeup (blog/video/slides): [https://your-writeup-link.example](https://your-writeup-link.example)
- Runnable training notebook: [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb)
- Runnable training script: [train_llm.py](train_llm.py)
- OpenEnv manifest: [openenv.yaml](openenv.yaml)

## Validation Checklist

- Public, cloneable Hugging Face Space link is present in this README.
- OpenEnv environment implements `reset()`, `step()`, and `state()` in [environment.py](environment.py).
- OpenEnv manifest is parseable and points to the environment class in [openenv.yaml](openenv.yaml).
- Training script and notebook are linked and runnable from this README.
- Training evidence images are committed and embedded below:
  - `docs/reward_curve.png`
  - `docs/loss_curve.png`

## 📈 Training & Results

We trained `[Insert Model Name Here, e.g., Llama-3.2-3B]` using **Group Relative Policy Optimization (GRPO)** via Hugging Face TRL directly against the Pricing Pro environment.

### Observable Improvements

**1. Reward Curve:** Shows the LLM learning to maximize profit over time.
![Reward Curve](docs/reward_curve.png)
*Caption: The agent starts by randomly pricing (often taking losses), but quickly learns to consistently find the optimal profit margin against the competitor.*

**2. Loss Curve:** Shows optimization stability across updates.
![Loss Curve](docs/loss_curve.png)

**3. Baseline Comparison:** 
![Baseline Comparison](docs/baseline_vs_trained.png)
*Caption: Untrained baseline vs. Trained AI-CFO in a highly price-sensitive market.*

---

## 🚀 Quickstart

### Prerequisites
```bash
pip install openenv[all] trl transformers
```

### Running the Environment Locally
```python
from openenv import make
import openenv

# Load the environment
env = make("pricing-pro-v1")
state = env.reset()

print("Market State:", state)

# Take a step (Agent sets prices for products)
next_state, reward, done, info = env.step({
    "prices": {
        "Premium_Watch": 189.0,
        "Standard_Earbuds": 44.0,
        "Budget_Cable": 7.5
    }
})

print("Reward generated:", reward)
print("Simulation Info:", info)
```

### Training Script
A complete, runnable notebook demonstrating the pipeline is available at [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb), and the script entrypoint is [train_llm.py](train_llm.py).

---

## 📚 References & Links
*   [Writeup (replace before submit)](https://your-writeup-link.example)
*   [Hugging Face Space (replace before submit)](https://huggingface.co/spaces/your-username/pricing-pro-ai-cfo)
*   [Training Notebook](notebooks/train_colab.ipynb)

> **Submission for the OpenEnv AI Hackathon**
> *Themes Addressed: Multi-Agent Interactions, Fleet AI (Scalable Oversight), Halluminate (Multi-Actor Environments).*