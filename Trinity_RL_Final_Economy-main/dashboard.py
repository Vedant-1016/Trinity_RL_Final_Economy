import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os
from environment import PricingProEnv

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pricing Pro: AI-CFO Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stChatFloatingInputContainer { bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'env' not in st.session_state:
    st.session_state.env = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_obs' not in st.session_state:
    st.session_state.current_obs = None
if 'last_feedback_write_ts' not in st.session_state:
    st.session_state.last_feedback_write_ts = 0.0

MAX_FEEDBACK_CHARS = 500
MIN_FEEDBACK_SAVE_INTERVAL_SEC = 5
RLHF_BUFFER_PATH = "rlhf_buffer.jsonl"


def _is_feedback_allowed_now():
    now = time.time()
    elapsed = now - st.session_state.last_feedback_write_ts
    return elapsed >= MIN_FEEDBACK_SAVE_INTERVAL_SEC


def _normalize_feedback(text):
    if not isinstance(text, str):
        return ""
    normalized = " ".join(text.split())
    return normalized[:MAX_FEEDBACK_CHARS].strip()

# --- SIDEBAR: SIMULATION PARAMETERS ---
with st.sidebar:
    st.title("🛒 Store Configuration")
    st.subheader("General Parameters")

    theme = st.selectbox("Store Theme", ["Tech Shop", "Fashion"])

    # Define products based on theme for non-hardcoded env init
    if theme == "Tech Shop":
        products = [
            {"name": "Premium_Watch", "cost": 150.0, "stock": 50},
            {"name": "Standard_Earbuds", "cost": 35.0, "stock": 200},
            {"name": "Budget_Cable", "cost": 5.0, "stock": 500}
        ]
    else:
        products = [
            {"name": "Designer_Handbag", "cost": 300.0, "stock": 30},
            {"name": "Leather_Boots", "cost": 80.0, "stock": 100},
            {"name": "Cotton_Tee", "cost": 15.0, "stock": 400}
        ]

    day_type = st.selectbox("Day Type", ["Normal Day", "Holiday Rush", "Payday Weekend", "Stormy Day"])
    vibe = st.selectbox("Neighborhood Vibe", ["Everyone's Spending", "People are Cautious", "Tech Boom"])
    traffic = st.multiselect("Target Customers", ["Students", "Professionals", "Families", "Tourists"], default=["Professionals"])
    rival = st.selectbox("Rival Strategy", ["The Price Cutter", "The Quality Leader", "Steady"])

    if st.button("🚀 Start/Reset Simulation", use_container_width=True):
        st.session_state.env = PricingProEnv(products=products, use_llm_consumers=True)
        demo_context = {
            "day_type": day_type,
            "vibe": vibe,
            "traffic": traffic,
            "rival": rival
        }
        st.session_state.current_obs = st.session_state.env.reset(demo_context=demo_context)
        st.session_state.history = []
        st.success("Simulation Initialized with LLM-Council Consumers!")

# --- AGENT INTERACTION HELPER ---
def get_ai_recommendation(obs):
    # Simulated model inference for the demo if 'final_pricing_pro_model' exists
    # If not, it uses a heuristic fallback
    return {p: round(v['cost'] * 1.25, 2) for p, v in obs['inventory_status'].items()} if 'inventory_status' in obs else {p: 100.0 for p in obs['inventory']}

# --- MAIN DASHBOARD ---
# ... (rest of main dashboard logic)

if st.session_state.env:
    env = st.session_state.env
    obs = st.session_state.current_obs
    
    # --- ROW 1: KEY METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Profit", f"${env.total_profit:.2f}")
    with col2:
        stock_sum = sum(v['stock'] for v in env.inventory.values())
        st.metric("Total Inventory", f"{stock_sum} units")
    with col3:
        st.metric("Day / Shift", f"{obs['day']} / {obs['shift']}")
    with col4:
        st.metric("Sentiment", f"{env.macro_params['consumer_sentiment']}")

    # --- ROW 2: LIVE CHAT & INVENTORY ---
    tab1, tab2, tab3 = st.tabs(["💬 Market Interaction", "📦 Inventory Detail", "📈 Sales Analytics"])

    with tab1:
        st.subheader("Live Scenario")
        # Display the Rich Scenario from the Council
        if 'scenario' in obs:
            st.write(f"### 📖 {obs['scenario']}")
        st.info(f"**Council Instruction:** {obs['instruction']}")
        st.json(obs['dna'])
        
        with st.container():
            st.write("### AI-CFO Price Update")
            c1, c2, c3 = st.columns(3)
            prices = {}
            for i, (prod, details) in enumerate(env.inventory.items()):
                cols = [c1, c2, c3]
                with cols[i % 3]:
                    cost = details['cost']
                    prices[prod] = st.number_input(f"{prod} (Cost: ${cost})", value=float(cost * 1.2), step=0.1, key=f"in_{prod}")

            if st.button("Get AI-CFO Recommendation"):
                rec_prices = get_ai_recommendation(obs)
                # Note: In Streamlit, updating number_input value via session_state requires specific handling
                # For this demo, we'll just display them or let the user see the log
                st.info(f"AI-CFO suggests: {rec_prices}")

            if st.button("Submit Prices & Run Shift"):
                next_obs, reward, done, info = env.step({"prices": prices})
                st.session_state.current_obs = next_obs
                st.session_state.history.append(info)
                
                if info['status'] == "Council Approved":
                    st.success(f"Sale Successful! Profit: ${info['shift_profit']}")
                else:
                    st.error(f"No Sale: {info['critique']}")
                
                if done:
                    st.warning("Simulation Complete! End of 7-day period.")
                    st.balloons()
        
        # --- RLHF FEEDBACK SECTION ---
        st.divider()
        st.subheader("✍️ Shopkeeper Coaching (RLHF)")
        feedback = st.text_area(
            "Provide feedback to the AI-CFO on this shift (e.g., 'Be more aggressive with student pricing'):",
            max_chars=MAX_FEEDBACK_CHARS,
        )
        if st.button("Save Feedback for Patch Update"):
            clean_feedback = _normalize_feedback(feedback)
            if not clean_feedback:
                st.warning("Feedback is empty after normalization. Please provide actionable text.")
            elif not _is_feedback_allowed_now():
                st.warning("Please wait a few seconds before submitting feedback again.")
            else:
                entry = {
                    "dna": st.session_state.current_obs.get("dna", {}),
                    "action": prices,
                    "feedback": clean_feedback,
                    "timestamp": time.time(),
                }
                with open(RLHF_BUFFER_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                st.session_state.last_feedback_write_ts = time.time()
                # Guard against unbounded growth in shared/public deployments.
                if os.path.exists(RLHF_BUFFER_PATH) and os.path.getsize(RLHF_BUFFER_PATH) > 5 * 1024 * 1024:
                    st.warning("RLHF buffer exceeded 5MB. Run patch update and archive/clear old feedback.")
                st.success("Feedback saved! It will be applied during the next Patch Update.")

    with tab2:
        st.subheader("Product Inventory Breakdown")
        df_inv = pd.DataFrame([
            {"Product": p, "Stock": s, "Cost": next(item['cost'] for item in env.products if item['name'] == p)}
            for p, s in env.inventory.items()
        ])
        st.table(df_inv)
        
        # Stock Level Chart
        fig_stock = px.bar(df_inv, x="Product", y="Stock", color="Product", title="Current Stock Levels")
        st.plotly_chart(fig_stock, use_container_width=True)

    with tab3:
        if st.session_state.history:
            st.subheader("Sales History & Price Logs")
            history_data = []
            for h in st.session_state.history:
                row = {"Status": h['status'], "Profit": h['shift_profit'], "Total": h['total_profit']}
                # Add competitor prices
                for p, price in h['competitor_prices'].items():
                    row[f"Comp_{p}"] = price
                history_data.append(row)
            
            df_hist = pd.DataFrame(history_data)
            st.dataframe(df_hist, use_container_width=True)
            
            # Profit Curve
            fig_profit = px.line(df_hist, y="Total", title="Profit Accumulation Over Time")
            st.plotly_chart(fig_profit, use_container_width=True)
        else:
            st.write("No sales data yet. Start the simulation!")

else:
    st.warning("Please configure your store in the sidebar and click 'Start Simulation'.")
    st.image("https://images.unsplash.com/photo-1441986300917-64674bd600d8?auto=format&fit=crop&q=80&w=1000", caption="Pricing Pro AI-CFO")

# --- FOOTER ---
st.markdown("---")
st.caption("Pricing Pro: OpenEnv Multi-Agent Market Simulation | RLHF Powered Demo")
