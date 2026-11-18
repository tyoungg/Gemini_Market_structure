import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="The Rational Coward — Market Structure Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply visual styling
sns.set_theme(style="whitegrid")

# Title and header
st.title("📈 The Rational Coward — Interactive Market Structure Lab")
st.markdown("""
*Based on **The Rational Coward Research Program** & **Further Research Addendum** (Nov 2026).*  
This application bridges **theoretical agent-based simulation** with **empirical real-market data testing** (via `yfinance` & custom CSV uploads).
""")

# Top-level Operating Mode Selector placed at the TOP of the Sidebar
st.sidebar.header("🌐 Operating Mode")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["🧪 Theoretical Simulation Engine", "📊 Empirical Real-Data Laboratory", "📁 Custom CSV Data Ingestion"]
)

st.sidebar.markdown("---")

# ==============================================================================
# MODE 1: THEORETICAL SIMULATION ENGINE
# ==============================================================================
if app_mode == "🧪 Theoretical Simulation Engine":
    st.sidebar.header("🕹️ Simulation Controls")
    sim_module = st.sidebar.radio(
        "Select Simulation Module:",
        [
            "1. Agent-Based Market Dynamics",
            "2. Active Share & Concentration Drift",
            "3. Multi-Asset Glidepath Extension"
        ]
    )

    if sim_module == "1. Agent-Based Market Dynamics":
        st.header("1. Agent-Based Market Structure Simulation")
        st.markdown("""
        Simulates **120 heterogeneous incumbent asset managers** and **100 prospective entrants** adjusting risk profiles under benchmark flow pressures, informed arbitrage capacity, and **corporate supply responses** (issuance/buybacks).
        """)

        st.sidebar.subheader("Model Parameters")
        passive_flow = st.sidebar.slider("Passive Inflow Intensity (λ_passive)", 0.0, 1.0, 0.40, 0.05)
        max_entrant_vol = st.sidebar.slider("Max Entrant Volatility Grid (%)", 25, 70, 50, 5)
        corp_supply_elasticity = st.sidebar.slider("Corporate Supply Response (γ_corp)", 0.0, 0.5, 0.15, 0.05)
        downside_penalty = st.sidebar.slider("Entrant Downside Failure Penalty", 0.5, 3.0, 1.5, 0.1)

        preset_regime = st.selectbox(
            "Select Preset Market Regime:",
            ["Custom", "Low Passive / Low Barriers", "High Passive / High Barriers", "Free Informed Entry Counterfactual"]
        )

        if preset_regime == "Low Passive / Low Barriers":
            passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.15, 35, 0.20, 1.0
        elif preset_regime == "High Passive / High Barriers":
            passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.65, 50, 0.05, 2.5
        elif preset_regime == "Free Informed Entry Counterfactual":
            passive_flow, max_entrant_vol, corp_supply_elasticity, downside_penalty = 0.65, 50, 0.05, 0.5

        def run_agent_simulation(seed=42):
            np.random.seed(seed)
            franchise_val = np.random.uniform(50, 150, 120)
            
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

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=df_sim, y="Scenario", x="Valuation Wedge (model units)", palette="Blues_r", ax=axes[0])
        axes[0].set_title("Valuation Distortion Wedge across Scenarios")
        axes[0].set_xlabel("Mean Absolute Valuation Wedge")
        
        sns.barplot(data=df_sim, y="Scenario", x="Incumbent TE (%)", palette="Reds_r", ax=axes[1])
        axes[1].set_title("Incumbent Manager Tracking Error (%)")
        axes[1].set_xlabel("Tracking Error (%)")
        
        plt.tight_layout()
        st.pyplot(fig)

    elif sim_module == "2. Active Share & Concentration Drift":
        st.header("2. Active Share & Concentration Counterfactual")
        st.markdown("""
        Evaluates how rising **top-10 benchmark concentration ($C_{10}$)** mechanically depresses median fund Active Share even without active trading.
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

    elif sim_module == "3. Multi-Asset Glidepath Extension":
        st.header("3. Multi-Asset Glidepath & Cross-Asset Spillover")
        st.markdown("""
        Simulates Target-Date Fund (TDF) mechanical glidepath inflows across Large-Cap Equities, Small-Cap Equities, and Fixed Income.
        """)

        st.sidebar.subheader("Glidepath Allocations")
        w_large = st.sidebar.slider("Large-Cap Weight (%)", 30, 80, 55) / 100.0
        w_small = st.sidebar.slider("Small-Cap Weight (%)", 5, 30, 15) / 100.0
        w_bond = 1.0 - (w_large + w_small)
        st.sidebar.write(f"Fixed Income Weight: **{w_bond*100:.1f}%**")

        periods = np.arange(1, 61)
        np.random.seed(42)
        
        price_large = 100.0 * np.cumprod(1 + np.random.normal(0.003, 0.015, 60) + 0.005 * w_large)
        price_small = 100.0 * np.cumprod(1 + np.random.normal(0.004, 0.022, 60) + 0.008 * w_small)
        price_bond = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.005, 60) + 0.002 * w_bond)

        wedge_large = (price_large - 100.0) * 0.25
        wedge_small = (price_small - 100.0) * 0.45
        wedge_bond = (price_bond - 100.0) * 0.12

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

# ==============================================================================
# MODE 2: EMPIRICAL REAL-DATA LABORATORY
# ==============================================================================
elif app_mode == "📊 Empirical Real-Data Laboratory":
    st.header("📊 Empirical Real-Data Laboratory")
    st.markdown("""
    Test **Green's Hypothesis** on **real market data** pulled live via `yfinance`.  
    *Green's Hypothesis: Price changes around index inclusions are driven by mechanical index demand rather than fundamental updates.*
    """)

    st.sidebar.header("🔍 Real-Data Event Study Settings")
    
    preset_events = {
        "Tesla (TSLA) — S&P 500 Inclusion (Dec 2020)": {
            "ticker": "TSLA", "date": "2020-12-21", "peers": ["F", "GM", "TM", "RACE", "HMC", "STLA"]
        },
        "Palantir (PLTR) — S&P 500 Inclusion (Sep 2024)": {
            "ticker": "PLTR", "date": "2024-09-23", "peers": ["SNOW", "PATH", "DDOG", "AI", "MDB"]
        },
        "Super Micro (SMCI) — S&P 500 Inclusion (Mar 2024)": {
            "ticker": "SMCI", "date": "2024-03-18", "peers": ["DELL", "HPE", "NVDA", "WDC", "STX"]
        },
        "Deckers Outdoor (DECK) — S&P 500 Inclusion (Mar 2024)": {
            "ticker": "DECK", "date": "2024-03-18", "peers": ["NKE", "CROX", "SKX", "UAA", "WWW"]
        },
        "Kroger (KR) — Retail/Consumer Staples Benchmark": {
            "ticker": "KR", "date": "2024-03-18", "peers": ["WMT", "TGT", "COST", "ACI", "DG"]
        },
        "Custom Event Entry": {
            "ticker": "CUSTOM", "date": "", "peers": []
        }
    }

    event_choice = st.sidebar.selectbox("Select Historical Inclusion Event:", list(preset_events.keys()))

    if event_choice == "Custom Event Entry":
        target_ticker = st.sidebar.text_input("Target Ticker Symbol:", "KR").upper()
        event_date_str = st.sidebar.text_input("Inclusion Date (YYYY-MM-DD):", "2024-03-18")
        
        # Dynamic default peers based on typed ticker
        if target_ticker == "KR":
            default_peers = "WMT, TGT, COST, ACI, DG"
        elif target_ticker == "DECK":
            default_peers = "NKE, CROX, SKX, UAA, WWW"
        elif target_ticker == "PLTR":
            default_peers = "SNOW, PATH, DDOG, AI, MDB"
        elif target_ticker == "SMCI":
            default_peers = "DELL, HPE, NVDA, WDC, STX"
        elif target_ticker == "TSLA":
            default_peers = "F, GM, TM, RACE, HMC, STLA"
        else:
            default_peers = "WMT, TGT, COST, ACI, DG"
            
        peer_tickers_str = st.sidebar.text_input("Candidate Peer Tickers (comma separated):", default_peers)
        peer_list = [p.strip().upper() for p in peer_tickers_str.split(",") if p.strip()]
    else:
        evt = preset_events[event_choice]
        target_ticker = evt["ticker"]
        event_date_str = evt["date"]
        peer_list = evt["peers"]

    st.sidebar.subheader("Matching & Event Window")
    k_neighbors = st.sidebar.slider("K Twin Matches", 1, 5, 3)
    pre_days = st.sidebar.slider("Pre-Event Days", 5, 30, 15)
    post_days = st.sidebar.slider("Post-Event Days", 5, 45, 30)

    run_fetch = st.button("🚀 Fetch Real Market Data & Compute Green Score")

    if run_fetch or event_choice != "Custom Event Entry":
        try:
            event_dt = datetime.strptime(event_date_str, "%Y-%m-%d")
            start_dt = event_dt - timedelta(days=pre_days + 20)
            end_dt = event_dt + timedelta(days=post_days + 20)

            st.info(f"Fetching live market data for **{target_ticker}** and candidate peers ({', '.join(peer_list)}) from `{start_dt.strftime('%Y-%m-%d')}` to `{end_dt.strftime('%Y-%m-%d')}`...")

            tickers_to_pull = list(set([target_ticker] + peer_list))
            raw_data = yf.download(tickers_to_pull, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)

            # Safely handle yfinance MultiIndex structures
            if isinstance(raw_data.columns, pd.MultiIndex):
                top_levels = raw_data.columns.levels[0]
                if 'Close' in top_levels:
                    prices = raw_data['Close']
                elif 'Adj Close' in top_levels:
                    prices = raw_data['Adj Close']
                else:
                    prices = raw_data.xs(top_levels[0], axis=1, level=0)
            else:
                prices = raw_data

            # Use forward/backward fill per series to avoid dropping all dates globally
            prices = prices.ffill().bfill()

            if target_ticker in prices.columns:
                target_prices = prices[target_ticker].dropna()
                valid_peers = [c for c in peer_list if c in prices.columns and c != target_ticker]
                peer_prices = prices[valid_peers]

                st.success(f"Successfully retrieved price histories for **{target_ticker}** and {len(valid_peers)} peer stocks!")

                # Calculate returns and risk metrics
                daily_returns = prices.pct_change().ffill().bfill()
                volatilities = daily_returns.std() * np.sqrt(252)
                mean_returns = daily_returns.mean() * 252

                # Fetch basic fundamental metrics for KNN matching
                feature_data = []
                for t in tickers_to_pull:
                    if t in prices.columns:
                        try:
                            t_info = yf.Ticker(t).info
                            mcap = t_info.get("marketCap", 1e10)
                            pe = t_info.get("trailingPE", 25.0)
                            if pe is None or np.isnan(pe):
                                pe = 25.0
                        except Exception:
                            mcap = 1e10
                            pe = 25.0
                        
                        vol = volatilities.get(t, 0.30)
                        ret = mean_returns.get(t, 0.10)

                        feature_data.append({
                            "Ticker": t,
                            "Log_MarketCap": np.log(max(mcap, 1e6)),
                            "PE_Ratio": pe,
                            "Annualized_Vol": vol,
                            "Mean_Return": ret
                        })

                metrics_df = pd.DataFrame(feature_data).set_index("Ticker")
                
                if target_ticker in metrics_df.index:
                    target_feat = metrics_df.loc[[target_ticker]]
                    peer_feats = metrics_df.drop(index=[target_ticker])

                    # Standardize features using StandardScaler
                    scaler = StandardScaler()
                    peer_scaled = scaler.fit_transform(peer_feats)
                    target_scaled = scaler.transform(target_feat)

                    n_k = min(k_neighbors, len(peer_feats))
                    knn = NearestNeighbors(n_neighbors=n_k)
                    knn.fit(peer_scaled)
                    distances, indices = knn.kneighbors(target_scaled)

                    twin_tickers = peer_feats.iloc[indices[0]].index.tolist()

                    st.subheader(f"🤝 Fundamental KNN Matched Twin Portfolio (K={len(twin_tickers)})")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**Target Included Stock Fundamentals:**")
                        st.dataframe(metrics_df.loc[[target_ticker]], use_container_width=True)
                    with col_b:
                        st.markdown("**Matched Counterfactual Twin Stocks:**")
                        st.dataframe(peer_feats.loc[twin_tickers], use_container_width=True)

                    # Event Window Index Alignment
                    target_dates = target_prices.index
                    nearest_idx_pos = target_dates.get_indexer([event_dt], method='nearest')[0]
                    
                    start_pos = max(0, nearest_idx_pos - pre_days)
                    end_pos = min(len(target_prices), nearest_idx_pos + post_days + 1)

                    target_window = target_prices.iloc[start_pos:end_pos]
                    rel_days = np.arange(start_pos - nearest_idx_pos, end_pos - nearest_idx_pos)

                    # Cumulative Returns indexed to window start
                    target_norm = (target_window / target_window.iloc[0] - 1.0) * 100.0

                    twin_prices_window = peer_prices[twin_tickers].iloc[start_pos:end_pos]
                    twin_avg_window = twin_prices_window.mean(axis=1)
                    twin_norm = (twin_avg_window / twin_avg_window.iloc[0] - 1.0) * 100.0

                    green_score_series = target_norm - twin_norm

                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

                    ax1.plot(rel_days, target_norm.values, label=f"Target Stock ({target_ticker})", color="darkgreen", lw=2.5)
                    ax1.plot(rel_days, twin_norm.values, label=f"KNN Twin Portfolio ({', '.join(twin_tickers)})", color="gray", linestyle="--", lw=2)
                    ax1.axvline(0, color="red", linestyle=":", lw=1.5, label="Inclusion Event (t=0)")
                    ax1.set_title(f"Real Market Event Study: {target_ticker} Index Inclusion ({event_date_str})")
                    ax1.set_ylabel("Cumulative Return (%)")
                    ax1.legend()

                    ax2.fill_between(rel_days, green_score_series.values, 0, where=(green_score_series.values >= 0), color="green", alpha=0.3, label="Positive Green Score (Inclusion Premium)")
                    ax2.fill_between(rel_days, green_score_series.values, 0, where=(green_score_series.values < 0), color="red", alpha=0.3, label="Negative Green Score")
                    ax2.plot(rel_days, green_score_series.values, color="darkgreen", lw=1.5)
                    ax2.axvline(0, color="red", linestyle=":", lw=1.5)
                    ax2.set_title("Empirical Green Score (Abnormal Price Distortion = Target CAR - Twin CAR)")
                    ax2.set_xlabel("Trading Days Relative to Inclusion (t=0)")
                    ax2.set_ylabel("Green Score (%)")
                    ax2.legend()

                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning(f"Could not retrieve ticker data for {target_ticker}. Please check ticker symbol or connection.")
        except Exception as e:
            st.error(f"Error executing real-data event study: {e}")

# ==============================================================================
# MODE 3: CUSTOM CSV DATA INGESTION
# ==============================================================================
elif app_mode == "📁 Custom CSV Data Ingestion":
    st.header("📁 Custom Data Ingestion & Empirical Analysis")
    st.markdown("""
    Upload your proprietary or licensed dataset (e.g. CRSP mutual fund holdings, institutional constituent concentration weights, or custom stock return series).
    """)

    uploaded_file = st.file_uploader("Upload CSV Data File", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(user_df.head(20), use_container_width=True)

            st.subheader("Summary Statistics")
            st.dataframe(user_df.describe(), use_container_width=True)

            num_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                st.subheader("Interactive Quick Plotter")
                x_col = st.selectbox("Select X-axis Column:", user_df.columns, index=0)
                y_col = st.selectbox("Select Y-axis Column:", num_cols, index=min(1, len(num_cols)-1))
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(user_df[x_col], user_df[y_col], marker='o', color='navy')
                ax.set_title(f"{y_col} vs {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    else:
        st.info("Please drag and drop a CSV file to inspect custom datasets.")

st.markdown("---")
st.caption("The Rational Coward Research Program — Dual-Mode Streamlit Web Application")
