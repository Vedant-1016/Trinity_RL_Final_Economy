# TECHNICAL_PIPELINE.md: Pricing Pro Exhaustive Architecture

This document provides a microscopic, exhaustive breakdown of every technical component, mathematical formula, and operational pipeline within the **Pricing Pro: AI-CFO Market Simulation**.

---

## 1. The Persona DNA Engine (`persona_engine.py`)
To train an AI-CFO properly, it must encounter a diverse, non-repeating array of customer psychologies. We replaced static buyer lists with a **5-Dimensional DNA Matrix**, yielding **576 unique combinations**.

### The 5 Dimensions:
1.  **Economic Profile (The "Wallet"):** Fixed/Low, Variable, Stable, High.
2.  **Psychological Driver (The "Why"):** Frugality, Performance, Status, Time-Sensitivity.
3.  **Market Intelligence (The "Awareness"):** Researcher, Loyalist, Skeptic, Impressionable.
4.  **Purchase Urgency (The "When"):** Predator, Just-in-Time, Panic-Buyer.
5.  **Volume & Context (The "Scale"):** Individual, Bulk, Gift-Giver.

### The Mechanism:
The `generate_random_dna()` function selects one trait from each dimension per interaction. This DNA tuple dictates both the textual narrative presented to the agent and the mathematical multipliers applied during the sales logic.

---

## 2. The Collaborative Council (`council.py`)
To prevent the SFT and RL phases from feeling synthetic, we built a **3-Model Collaborative Council** utilizing the Groq API. 

### The Tri-Model Loop:
1.  **Llama 3.1 70B (The Narrator):** Receives the DNA and creates a base 2-sentence scenario.
2.  **Mixtral 8x7B (The Analyst):** Reviews the scenario and injects a psychological quirk consistent with the DNA.
3.  **Llama 3.1 8B (The Synthesizer):** Merges the outputs into a polished, label-free narrative.

*Usage:* This Council generates the SFT **golden** samples (default **5** via `SFT_SAMPLES` / `generate_sft_data.py`), produces dynamic scenarios during the live demo, and acts as the "Social Judge" during the council online-RL phase (default **10** scenarios).

---

## 3. The Live Environment & Physics (`environment.py`)
The environment preserves the original **probabilistic market physics** while removing all hardcoded variables.

### The 9 Macro Parameters (Dynamic Ranges):
*   `inflation_rate` (0.01 - 0.15)
*   `consumer_sentiment` (0.1 - 1.0)
*   `supply_chain_disruption` (0.0 - 1.0)
*   `competitor_aggressiveness` (0.5 - 1.5)
*   `seasonal_demand_multiplier` [0.8, 1.0, 1.5, 2.0]
*   `market_growth_rate` (-0.05 - 0.10)
*   `price_sensitivity_index` (0.5 - 2.0)
*   `brand_preference_bias` (0.5 - 2.0)
*   `purchase_urgency` (0.5 - 2.0)

### The Probabilistic Sales Formula:
During the `step()` function, if the Council "Negotiation" does not trigger a hard rejection (e.g., pricing below cost or >3x competitor), the exact market physics take over.

1.  **DNA Modifiers:** The base `price_sensitivity_index` and `purchase_urgency` are multiplied by DNA traits (e.g., a "Researcher" multiplies sensitivity by 1.5; a "Panic-Buyer" multiplies urgency by 2.0).
2.  **The Calculus:**
    ```python
    our_value = brand_score / (ai_price ** price_penalty)
    comp_value = 1.0 / (competitor_price ** price_penalty)
    score_diff = our_value - comp_value
    x = max(min(5.0 * score_diff, 10), -10)
    buy_probability = 1.0 / (1.0 + math.exp(-x))
    ```
3.  **The Basket Logic:** The AI must manage multiple products simultaneously. The success of the "shift" requires at least 50% of the basket to be sold, encouraging complex strategies like loss-leading.

---

## 4. The Reviewer Engine (`reviewer_engine.py`)
This engine bridges the mathematical environment with the LLM training pipeline by converting physics into specific text critiques and reward signals.

### The Reward / Penalty Schema:
*   **Success (Reward):** `Profit Margin % * 10.0`. Anchors the model to maximize profit.
*   **Hard Penalty:** `-20.0` for pricing below cost.
*   **DNA Penalty:** `-10.0` for violating a hard budget ceiling (Economic: Fixed/Low).
*   **Competitor Penalty:** `-5.0` for losing a "Researcher" to the rival.
*   **Soft Penalty:** `-2.0` for general probabilistic failure.

*Critiques:* Each failure appends a specific textual reason (e.g., "Researcher found cheaper price at rival") which the Agent uses in its next loop.

---

## 5. The Training Pipeline (`train_llm.py`)
Designed to fine-tune an 8B or 3B model (e.g., `Llama-3.2-3B`) within 24 hours.

### Phase 1: SFT Bootstrap
Using `sft_dataset.json` (from `generate_sft_data.py`), SFT training runs a **short** `max_steps` run scaled to `SFT_SAMPLES` (default **5** samples). This warms up JSON (`{"prices": ...}`) and margin structure.

### Phase 2: Online RL (heuristic) — 50 **outer** scenarios
`HEURISTIC_SCENARIOS` is **50** in code defaults (not 2000/4000). The Math Heuristic may use up to **3 inner attempts** per scenario, but `training_metrics.json` stores **one row per outer scenario** — **50 heuristic rows total**, not 50×3.
*   **The 3-Loop Structure:** For each `i` in `0..49`, the Agent has up to 3 tries; early exit on success.
*   **Error-Driven Backprop** on each inner step that runs.
*   **One metrics row** after the inner work for that `i` (last attempt’s values).

### Phase 3: Council online RL — 10 scenarios
`COUNCIL_SCENARIOS` default is **10**. Same 3-loop pattern; **one metrics row per council scenario**. The Agent faces the council `council_review` (Groq-backed when configured) instead of the fast heuristic.

---

## 6. The live demo (web + notebook)

The **public-facing product UI** is the Vite / React app under `Trinity_RL_Final_Economy-main/` (deploy e.g. to Vercel) with the marketing home page, NAFO console, and embedded dashboard sections.

*   **Repro training:** `colab/quick_train_pricing_pro.ipynb` runs a small 3-phase training path with the user’s own `GROQ_API_KEY`, writes `training_metrics.json`, and exports plots to `docs/`. See `colab/README.md`.

*   The legacy **Streamlit** dashboard and **post-demo `patch_update.py` RLHF** loop have been **removed** to avoid duplicate UIs; training remains fully covered by `train_llm.py`, `tools/run_long_training.py`, and the Colab notebook.
