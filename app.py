import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import scipy.special

# Set page configuration
st.set_page_config(
    page_title="The Rational Coward — Market Structure Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply styling
sns.set_theme(style="whitegrid")

# Title and header
st.title("📈 The Rational Coward — Interactive Market Structure Lab")
st.markdown("""
*Based on **The Rational Coward Research Program** & **Further Research Addendum** (Nov 2026).*  
This application simulates how passive index flows, active manager risk constraints, corporate supply responses, and multi-asset glidepaths impact asset valuation and price discovery.
""")

# Sidebar Navigation
st.sidebar.header("🕹️ Simulation Controls")
module_choice = st.sidebar.radio(
    "Select Research Module:",
    [
        "1. Agent-Based Market Simulation",
        "2. Twin Matching & Green Score",
        "3. Active Share & Concentration Drift",
        "4. Multi-Asset Glidepath Extension"
    ]
)

# ==========================================
# MODULE 1: AGENT-BASED SIMULATION
# ==========================================
if module_choice == "1. Agent-Based Market Simulation":
    st.header("1. Agent-Based Market Structure Simulation")
    st.markdown("""
    Explore how **120 heterogeneous incumbent asset managers** and **100 prospective entrants** adjust their tracking error and active risk under benchmark flow pressure.
    """)

    col1, col2, col3 = st.sidebar.columns(2), st.sidebar.columns(1), st.sidebar.columns(1)
    
    st.sidebar.subheader("Model Parameters")
    passive_flow = st.sidebar.slider("Passive Inflow Intensity (λ_passive)", 0.0, 1.0, 0.40, 0.05)
    max_entrant_vol = st.sidebar.slider("Max Entrant Volatility Grid (%)", 25, 70, 50, 5)
    corp_supply_elasticity = st.sidebar.slider("Corporate Supply Response (γ_corp)", 0.0, 0.5, 0.15, 0.05)
    downside_penalty = st.sidebar.slider("Entrant Downside Failure Penalty", 0.5, 3.0, 1.5, 0.1)
    
    preset_regime = st.selectbox(
        "Select Quick Preset Market Regime:",
        ["Custom", "Low Passive / Low Barriers", "High Passive / High Barriers", "Free Informed Entry Counterfactual"]
    )

    if preset_regime == "Low Passive / Low Barriers":
        passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.15, 35, 0.20, 1.0
    elif preset_regime == "High Passive / High Barriers":
        passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.65, 50, 0.05, 2.5
    elif preset_regime == "Free Informed Entry Counterfactual":
        passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.65, 50, 0.05, 0.5

    # Simulation Function
    def run_agent_simulation(n_incumbents=120, n_entrants=100, seed=42):
        np.random.seed(seed)
        franchise_val = np.random.uniform(50, 150, n_incumbents)
        
        scenarios = [
            ("Low passive / low barriers", 0.15, 0.70, 0.10, 25),
            ("High passive / low barriers", 0.65, 0.20, 0.10, 25),
            ("Low passive / high barriers", 0.15, 0.25, 0.05, 25),
            ("High passive / high barriers (no corp response)", 0.65, 0.05, 0.00, 25),
            ("High passive / high barriers (+ Corporate Supply)", passive_flow, 0.05, corp_supply_elasticity, max_entrant_vol),
            ("High passive / high barriers / free informed entry", passive_flow, 0.70, corp_supply_elasticity, max_entrant_vol)
        ]
        
        results = []
        for name, p_flow, inf_cap, c_supp, m_vol in scenarios:
            inc_te = 5.0 * np.exp(-2.2 * p_flow) / (1 + 0.01 * np.mean(franchise_val))
            
            vol_grid = np.linspace(4, m_vol, 10)
            success_prob = 1.0 / (1.0 + np.exp(-(vol_grid - 15.0) / 5.0))
            downside_risk = (vol_grid / m_vol) ** downside_penalty
            payoff = success_prob * 10.0 - downside_risk * 5.0
            best_ent_vol = vol_grid[np.argmax(payoff)]
            
            raw_wedge = (p_flow * 25.0) / (inf_cap + 0.1)
            adj_wedge = raw_wedge / (1.0 + 8.0 * c_supp)
            
            results.append({
                "Scenario": name,
                "Incumbent TE (%)": round(inc_te, 2),
                "Entrant Risk Choice (%)": round(best_ent_vol, 1),
                "Informed Capacity": round(inf_cap, 2),
                "Corp Supply Elasticity": round(c_supp, 2),
                "Valuation Wedge (model units)": round(adj_wedge, 2)
            })
            
        return pd.DataFrame(results)

    df_sim = run_agent_simulation()
    
    st.subheader("Simulation Results Table")
    st.dataframe(df_sim, use_container_width=True)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(data=df_sim, y="Scenario", x="Valuation Wedge (model units)", palette="Blues_r", ax=axes[0])
    axes[0].set_title("Valuation Distortion Wedge across Scenarios")
    axes[0].set_xlabel("Mean Absolute Valuation Wedge")
    
    sns.barplot(data=df_sim, y="Scenario", x="Incumbent TE (%)", palette="Reds_r", ax=axes[1])
    axes[1].set_title("Incumbent Manager Tracking Error (%)")
    axes[1].set_xlabel("Tracking Error (%)")
    
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================
# MODULE 2: TWIN MATCHING & GREEN SCORE
# ==========================================
elif module_choice == "2. Twin Matching & Green Score":
    st.header("2. Market Structure Lab: Twin Matching & Green Score")
    st.markdown("""
    Isolate mechanical index inclusion demand from underlying fundamental shifts using **K-Nearest Neighbors (KNN)** matched twin portfolios.
    """)

    st.sidebar.subheader("Matching Parameters")
    n_twins = st.sidebar.slider("Number of Twin Matches (K)", 1, 5, 3)
    announcement_pump = st.sidebar.slider("Index Inclusion Demand Pump (%)", 1.0, 10.0, 4.5, 0.5)

    @st.cache_data
    def generate_universe():
        np.random.seed(123)
        tickers = [f"UNIV_{i:03d}" for i in range(300)]
        mcap = np.random.lognormal(22, 0.8, 300)
        pe = np.random.normal(25, 8, 300)
        rev = np.random.normal(0.12, 0.05, 300)
        vol = np.random.normal(0.30, 0.08, 300)
        return pd.DataFrame({
            "Ticker": tickers,
            "MarketCap": mcap,
            "PE_Ratio": pe,
            "RevGrowth": rev,
            "Volatility": vol
        })

    df_univ = generate_universe()
    target_stock = pd.DataFrame([{
        "Ticker": "TARGET_INC",
        "MarketCap": df_univ["MarketCap"].median(),
        "PE_Ratio": 27.5,
        "RevGrowth": 0.14,
        "Volatility": 0.32
    }])

    # KNN Fit
    features = ["MarketCap", "PE_Ratio", "RevGrowth", "Volatility"]
    X = df_univ[features].values
    X_target = target_stock[features].values
    
    # Scale
    mean, std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - mean) / std
    X_target_scaled = (X_target - mean) / std

    knn = NearestNeighbors(n_neighbors=n_twins)
    knn.fit(X_scaled)
    distances, indices = knn.kneighbors(X_target_scaled)
    twins_df = df_univ.iloc[indices[0]]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Included Target Stock")
        st.dataframe(target_stock)
    with col2:
        st.subheader(f"Matched Counterfactual Twin Portfolio (K={n_twins})")
        st.dataframe(twins_df)

    # Event study chart
    event_days = np.arange(-10, 21)
    np.random.seed(999)
    target_ret = np.random.normal(0.0005, 0.012, len(event_days))
    target_ret[10:13] += (announcement_pump / 100.0) / 3.0
    twin_ret = np.random.normal(0.0004, 0.010, len(event_days))
    
    target_car = np.cumsum(target_ret) * 100
    twin_car = np.cumsum(twin_ret) * 100
    green_score = target_car - twin_car

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(event_days, target_car, label="Target Included Stock", color="darkgreen", lw=2)
    ax.plot(event_days, twin_car, label=f"KNN Twin Portfolio (K={n_twins})", color="gray", linestyle="--", lw=2)
    ax.fill_between(event_days, target_car, twin_car, color="green", alpha=0.15, label="Green Score (Abnormal Distortion)")
    ax.axvline(0, color="red", linestyle=":", label="Inclusion Announcement (t=0)")
    ax.set_title("Event Window Cumulative Abnormal Return (CAR)")
    ax.set_xlabel("Days Relative to Inclusion Announcement")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    st.pyplot(fig)

# ==========================================
# MODULE 3: ACTIVE SHARE DRIFT
# ==========================================
elif module_choice == "3. Active Share & Concentration Drift":
    st.header("3. Active Share & Concentration Counterfactual")
    st.markdown("""
    Evaluate how rising top-10 index concentration ($C_{10}$) mechanically depresses median fund Active Share without manager trade execution.
    """)

    years = np.arange(2010, 2026)
    c10_start = st.sidebar.slider("Starting Top-10 Concentration (%)", 15.0, 25.0, 18.0) / 100.0
    c10_end = st.sidebar.slider("Ending Top-10 Concentration (%)", 25.0, 45.0, 35.0) / 100.0

    c10_series = np.linspace(c10_start, c10_end, len(years))
    active_share_actual = 68.0 - 110.0 * (c10_series - c10_start) + np.random.normal(0, 0.8, len(years))
    active_share_notrade = 68.0 - 95.0 * (c10_series - c10_start)

    df_drift = pd.DataFrame({
        "Year": years,
        "Top-10 Concentration (C10)": c10_series,
        "Actual Median Active Share (%)": active_share_actual,
        "No-Trade Drift Counterfactual (%)": active_share_notrade
    })

    fig, ax1 = plt.subplots(figsize=(10, 4))
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Top-10 Concentration (C10)', color=color)
    ax1.plot(years, c10_series, color=color, marker='o', linewidth=2, label="C10 Concentration")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Active Share (%)', color=color)
    ax2.plot(years, active_share_actual, color='crimson', linestyle='-', label='Observed Active Share')
    ax2.plot(years, active_share_notrade, color='orange', linestyle='--', label='No-Trade Mechanical Drift')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Mechanical Active Share Drift vs. Top-10 Index Concentration")
    fig.tight_layout()
    st.pyplot(fig)

# ==========================================
# MODULE 4: MULTI-ASSET EXTENSION
# ==========================================
elif module_choice == "4. Multi-Asset Glidepath Extension":
    st.header("4. Multi-Asset Glidepath & Cross-Asset Spillover")
    st.markdown("""
    Simulate Target-Date Fund (TDF) mechanical glidepath inflows across Large-Cap Equities, Small-Cap Equities, and Fixed Income.
    """)

    st.sidebar.subheader("Glidepath Allocations")
    w_large = st.sidebar.slider("Large-Cap Weight (%)", 30, 80, 55) / 100.0
    w_small = st.sidebar.slider("Small-Cap Weight (%)", 5, 30, 15) / 100.0
    w_bond = 1.0 - (w_large + w_small)
    st.sidebar.write(f"Fixed Income Weight: **{w_bond*100:.1f}%**")

    rebal_trigger = st.sidebar.slider("TDF Rebalance Band (±%)", 1.0, 5.0, 3.0) / 100.0

    periods = np.arange(1, 61)
    np.random.seed(42)
    
    price_large = 100.0 * np.cumprod(1 + np.random.normal(0.003, 0.015, 60) + 0.005 * w_large)
    price_small = 100.0 * np.cumprod(1 + np.random.normal(0.004, 0.022, 60) + 0.008 * w_small)
    price_bond = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.005, 60) + 0.002 * w_bond)

    wedge_large = (price_large - 100.0) * 0.25
    wedge_small = (price_small - 100.0) * 0.45
    wedge_bond = (price_bond - 100.0) * 0.12

    df_multi = pd.DataFrame({
        "Period": periods,
        "Large-Cap Price": price_large,
        "Small-Cap Price": price_small,
        "Bond Index Price": price_bond,
        "Large-Cap Wedge (%)": wedge_large,
        "Small-Cap Wedge (%)": wedge_small,
        "Bond Wedge (%)": wedge_bond
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(periods, price_large, label="Large-Cap Equities (S&P 500)", color="navy")
    ax1.plot(periods, price_small, label="Small-Cap Equities (Russell 2000)", color="darkgreen")
    ax1.plot(periods, price_bond, label="Aggregate Bond Index", color="purple")
    ax1.set_title("Multi-Asset Price Trajectories under Glidepath Inflows")
    ax1.set_xlabel("Periods (Months)")
    ax1.set_ylabel("Asset Price Index")
    ax1.legend()

    ax2.plot(periods, wedge_large, label="Large-Cap Wedge (%)", color="navy")
    ax2.plot(periods, wedge_small, label="Small-Cap Wedge (%)", color="darkgreen", linestyle="--")
    ax2.plot(periods, wedge_bond, label="Bond Wedge (%)", color="purple", linestyle=":")
    ax2.set_title("Cross-Asset Valuation Distortion Wedges (%)")
    ax2.set_xlabel("Periods (Months)")
    ax2.set_ylabel("Valuation Distortion (%)")
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("The Rational Coward Research Program — Interactive Streamlit Web Application")
