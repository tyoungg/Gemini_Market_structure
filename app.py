import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import os

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
            "3. Multi-Asset Glidepath Extension",
            "4. Corporate Supply & Multi-Asset Elasticity"
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

    elif sim_module == "4. Corporate Supply & Multi-Asset Elasticity":
        st.header("4. Corporate Supply Response & Multi-Asset Elasticity (γ_corp)")
        st.markdown("""
        Evaluates how **Corporate Supply Responses** (Gross Share Buybacks vs. New Share Issuance) dynamically offset or absorb mechanical passive index flow pressures across asset classes.
        """)

        st.sidebar.subheader("Elasticity Parameters")
        gamma_large_input = st.sidebar.slider("Large-Cap Supply Response (γ_large)", 0.01, 0.40, 0.15, 0.01)
        gamma_small_input = st.sidebar.slider("Small-Cap Supply Response (γ_small)", 0.01, 0.30, 0.06, 0.01)
        gamma_bond_input = st.sidebar.slider("Bond Market Supply Response (γ_bond)", 0.01, 0.50, 0.22, 0.01)

        quarters = [f"{y}-Q{q}" for y in range(2015, 2026) for q in range(1, 5)]
        np.random.seed(42)
        n_q = len(quarters)

        p_intensity = np.linspace(0.20, 0.55, n_q)
        
        w_lc = (p_intensity * 22.0) / (1.0 + 8.0 * gamma_large_input) + np.random.normal(0, 0.2, n_q)
        w_sc = (p_intensity * 30.0) / (1.0 + 8.0 * gamma_small_input) + np.random.normal(0, 0.3, n_q)
        w_fi = (p_intensity * 12.0) / (1.0 + 8.0 * gamma_bond_input) + np.random.normal(0, 0.1, n_q)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Valuation Wedge Trajectories Across Asset Classes")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(quarters[::2], w_lc[::2], label="Large-Cap Equities", color="navy", marker='o')
            ax.plot(quarters[::2], w_sc[::2], label="Small/Mid-Cap Equities", color="darkgreen", marker='s', linestyle='--')
            ax.plot(quarters[::2], w_fi[::2], label="Fixed Income Debt", color="purple", marker='^', linestyle=':')
            ax.set_xticks(quarters[::4])
            ax.set_xticklabels(quarters[::4], rotation=45)
            ax.set_ylabel("Valuation Distortion Wedge (%)")
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Elasticity Dampening Summary")
            summary_df = pd.DataFrame({
                "Asset Class": ["Large-Cap Equities", "Small/Mid-Cap Equities", "Corporate Debt Index"],
                "Supply Elasticity (γ)": [gamma_large_input, gamma_small_input, gamma_bond_input],
                "Absorptive Capacity": ["High (Heavy Buybacks)", "Low (Net Issuance/Secondary Offerings)", "Very High (Flexible Debt Issuance)"],
                "Average Wedge (%)": [round(w_lc.mean(), 2), round(w_sc.mean(), 2), round(w_fi.mean(), 2)]
            })
            st.dataframe(summary_df, use_container_width=True)

# ==============================================================================
# MODE 2: EMPIRICAL REAL-DATA LABORATORY
# ==============================================================================
elif app_mode == "📊 Empirical Real-Data Laboratory":
    st.header("📊 Empirical Real-Data Laboratory")
    st.markdown("""
    Test **Green's Hypothesis** on **real market data** pulled live via `yfinance`.  
    *Green's Hypothesis: Price changes around index inclusions are driven by mechanical index demand rather than fundamental updates.*
    """)

    st.sidebar.header("🔍 Event Study Settings")
    
    study_type = st.sidebar.radio(
        "Select Event Study Mode:",
        ["🎯 Single Inclusion Event Study", "⚔️ Compare ANY Two Events (Head-to-Head)"]
    )

    preset_catalog = {
        "Tesla (TSLA) — S&P 500 Inclusion (Dec 2020)": {"ticker": "TSLA", "date": "2020-12-21", "peers": ["F", "GM", "TM", "RACE", "HMC", "STLA"]},
        "Palantir (PLTR) — S&P 500 Inclusion (Sep 2024)": {"ticker": "PLTR", "date": "2024-09-23", "peers": ["SNOW", "PATH", "DDOG", "AI", "MDB"]},
        "Super Micro (SMCI) — S&P 500 Inclusion (Mar 2024)": {"ticker": "SMCI", "date": "2024-03-18", "peers": ["DELL", "HPE", "NVDA", "WDC", "STX"]},
        "Deckers Outdoor (DECK) — S&P 500 Inclusion (Mar 2024)": {"ticker": "DECK", "date": "2024-03-18", "peers": ["NKE", "CROX", "SKX", "UAA", "WWW"]},
        "Kroger (KR) — Retail Benchmark": {"ticker": "KR", "date": "2024-03-18", "peers": ["WMT", "TGT", "COST", "ACI", "DG"]},
        "Custom Event Entry": {"ticker": "CUSTOM", "date": "2024-01-01", "peers": []}
    }

    st.sidebar.subheader("Matching & Event Window")
    k_neighbors = st.sidebar.slider("K Twin Matches", 1, 10, 3)
    pre_days = st.sidebar.slider("Pre-Event Days", 5, 365, 30)
    post_days = st.sidebar.slider("Post-Event Days", 5, 365, 60)

    def fetch_event_green_score(target_ticker, event_date_str, peer_list, k_n, pre_d, post_d):
        try:
            event_dt = datetime.strptime(event_date_str, "%Y-%m-%d")
            start_dt = event_dt - timedelta(days=pre_d + 30)
            end_dt = event_dt + timedelta(days=post_d + 30)

            tickers_to_pull = list(set([target_ticker] + peer_list))
            raw_data = yf.download(tickers_to_pull, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)

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

            prices = prices.ffill().bfill()

            if target_ticker not in prices.columns:
                return None, f"Could not fetch ticker {target_ticker}"

            target_prices = prices[target_ticker].dropna()
            valid_peers = [c for c in peer_list if c in prices.columns and c != target_ticker]
            peer_prices = prices[valid_peers]

            daily_returns = prices.pct_change().ffill().bfill()
            volatilities = daily_returns.std() * np.sqrt(252)
            mean_returns = daily_returns.mean() * 252

            feature_data = []
            for t in tickers_to_pull:
                if t in prices.columns:
                    try:
                        t_info = yf.Ticker(t).info
                        mcap = t_info.get("marketCap", 1e10) if isinstance(t_info, dict) else 1e10
                        pe = t_info.get("trailingPE", 25.0) if isinstance(t_info, dict) else 25.0
                        if mcap is None or not isinstance(mcap, (int, float)) or np.isnan(mcap) or mcap <= 0:
                            mcap = 1e10
                        if pe is None or not isinstance(pe, (int, float)) or np.isnan(pe) or pe <= 0:
                            pe = 25.0
                    except Exception:
                        mcap = 1e10
                        pe = 25.0
                    
                    vol = volatilities.get(t, 0.30)
                    ret = mean_returns.get(t, 0.10)
                    if pd.isna(vol) or np.isinf(vol): vol = 0.30
                    if pd.isna(ret) or np.isinf(ret): ret = 0.10

                    feature_data.append({
                        "Ticker": t,
                        "Log_MarketCap": np.log(max(mcap, 1e6)),
                        "PE_Ratio": pe,
                        "Annualized_Vol": vol,
                        "Mean_Return": ret
                    })

            metrics_df = pd.DataFrame(feature_data).set_index("Ticker").replace([np.inf, -np.inf], np.nan)
            for col in metrics_df.columns:
                if metrics_df[col].isnull().any():
                    col_med = metrics_df[col].median()
                    metrics_df[col] = metrics_df[col].fillna(col_med if not pd.isna(col_med) else 0.0)

            target_feat = metrics_df.loc[[target_ticker]]
            peer_feats = metrics_df.drop(index=[target_ticker])

            if len(peer_feats) == 0:
                return None, "No valid peer stocks for KNN matching."

            scaler = StandardScaler()
            peer_scaled = scaler.fit_transform(peer_feats)
            target_scaled = scaler.transform(target_feat)

            n_k = min(k_n, len(peer_feats))
            knn = NearestNeighbors(n_neighbors=n_k)
            knn.fit(peer_scaled)
            distances, indices = knn.kneighbors(target_scaled)

            twin_tickers = peer_feats.iloc[indices[0]].index.tolist()

            target_dates = target_prices.index
            nearest_idx_pos = target_dates.get_indexer([event_dt], method='nearest')[0]
            
            start_pos = max(0, nearest_idx_pos - pre_d)
            end_pos = min(len(target_prices), nearest_idx_pos + post_d + 1)

            target_window = target_prices.iloc[start_pos:end_pos]
            rel_days = np.arange(start_pos - nearest_idx_pos, end_pos - nearest_idx_pos)

            target_norm = (target_window / target_window.iloc[0] - 1.0) * 100.0

            twin_prices_window = peer_prices[twin_tickers].iloc[start_pos:end_pos]
            twin_avg_window = twin_prices_window.mean(axis=1)
            twin_norm = (twin_avg_window / twin_avg_window.iloc[0] - 1.0) * 100.0

            green_score_series = target_norm - twin_norm

            res_dict = {
                "rel_days": rel_days,
                "target_norm": target_norm,
                "twin_norm": twin_norm,
                "green_score": green_score_series,
                "target_ticker": target_ticker,
                "event_date": event_date_str,
                "twin_tickers": twin_tickers,
                "metrics_df": metrics_df,
                "peer_feats": peer_feats
            }
            return res_dict, None
        except Exception as e:
            return None, str(e)

    # --------------------------------------------------------------------------
    # OPTION A: SINGLE INCLUSION EVENT STUDY
    # --------------------------------------------------------------------------
    if study_type == "🎯 Single Inclusion Event Study":
        event_choice = st.sidebar.selectbox("Select Inclusion Event:", list(preset_catalog.keys()))

        if event_choice == "Custom Event Entry":
            target_ticker = st.sidebar.text_input("Target Ticker:", "TSLA").upper()
            event_date_str = st.sidebar.text_input("Inclusion Date (YYYY-MM-DD):", "2020-12-21")
            peer_tickers_str = st.sidebar.text_input("Peer Tickers (comma separated):", "F, GM, TM, RACE, HMC")
            peer_list = [p.strip().upper() for p in peer_tickers_str.split(",") if p.strip()]
        else:
            evt = preset_catalog[event_choice]
            target_ticker = evt["ticker"]
            event_date_str = evt["date"]
            peer_list = evt["peers"]

        run_single = st.button("🚀 Fetch Data & Compute Single Event Green Score")

        if run_single or event_choice != "Custom Event Entry":
            res, err = fetch_event_green_score(target_ticker, event_date_str, peer_list, k_neighbors, pre_days, post_days)
            if err:
                st.error(f"Error executing study: {err}")
            else:
                st.success(f"Successfully calculated Green Score for **{target_ticker}** around {event_date_str}!")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Target Stock Fundamentals:**")
                    st.dataframe(res["metrics_df"].loc[[target_ticker]], use_container_width=True)
                with col_b:
                    st.markdown(f"**Matched Twin Stocks (K={len(res['twin_tickers'])}):**")
                    st.dataframe(res["peer_feats"].loc[res["twin_tickers"]], use_container_width=True)

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

                ax1.plot(res["rel_days"], res["target_norm"].values, label=f"Target ({target_ticker})", color="darkgreen", lw=2.5)
                ax1.plot(res["rel_days"], res["twin_norm"].values, label=f"Twin Portfolio ({', '.join(res['twin_tickers'])})", color="gray", linestyle="--", lw=2)
                ax1.axvline(0, color="red", linestyle=":", lw=1.5, label="Inclusion Date (t=0)")
                ax1.set_title(f"Event Study: {target_ticker} Index Inclusion ({event_date_str})")
                ax1.set_ylabel("Cumulative Return (%)")
                ax1.legend()

                ax2.fill_between(res["rel_days"], res["green_score"].values, 0, where=(res["green_score"].values >= 0), color="green", alpha=0.3, label="Positive Green Score (Inclusion Premium)")
                ax2.fill_between(res["rel_days"], res["green_score"].values, 0, where=(res["green_score"].values < 0), color="red", alpha=0.3, label="Negative Green Score")
                ax2.plot(res["rel_days"], res["green_score"].values, color="darkgreen", lw=1.5)
                ax2.axvline(0, color="red", linestyle=":", lw=1.5)
                ax2.set_title("Empirical Green Score (Target CAR - Twin CAR)")
                ax2.set_xlabel("Trading Days Relative to Inclusion (t=0)")
                ax2.set_ylabel("Green Score (%)")
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

    # --------------------------------------------------------------------------
    # OPTION B: DUAL EVENT HEAD-TO-HEAD COMPARISON (ANY 2 STOCKS)
    # --------------------------------------------------------------------------
    else:
        st.sidebar.subheader("⚔️ Select Stock #1")
        ev1_choice = st.sidebar.selectbox("Stock #1 Preset:", list(preset_catalog.keys()), index=0)
        if ev1_choice == "Custom Event Entry":
            t1 = st.sidebar.text_input("Stock #1 Ticker:", "TSLA").upper()
            d1 = st.sidebar.text_input("Stock #1 Inclusion Date:", "2020-12-21")
            p1_str = st.sidebar.text_input("Stock #1 Peers:", "F, GM, TM, RACE, HMC")
            p1 = [p.strip().upper() for p in p1_str.split(",") if p.strip()]
        else:
            evt1 = preset_catalog[ev1_choice]
            t1, d1, p1 = evt1["ticker"], evt1["date"], evt1["peers"]

        st.sidebar.subheader("⚔️ Select Stock #2")
        ev2_choice = st.sidebar.selectbox("Stock #2 Preset:", list(preset_catalog.keys()), index=1)
        if ev2_choice == "Custom Event Entry":
            t2 = st.sidebar.text_input("Stock #2 Ticker:", "PLTR").upper()
            d2 = st.sidebar.text_input("Stock #2 Inclusion Date:", "2024-09-23")
            p2_str = st.sidebar.text_input("Stock #2 Peers:", "SNOW, PATH, DDOG, AI, MDB")
            p2 = [p.strip().upper() for p in p2_str.split(",") if p.strip()]
        else:
            evt2 = preset_catalog[ev2_choice]
            t2, d2, p2 = evt2["ticker"], evt2["date"], evt2["peers"]

        run_dual = st.button("⚔️ Fetch & Compare ANY Two Events Head-to-Head")

        if run_dual or True:
            res1, err1 = fetch_event_green_score(t1, d1, p1, k_neighbors, pre_days, post_days)
            res2, err2 = fetch_event_green_score(t2, d2, p2, k_neighbors, pre_days, post_days)

            if err1:
                st.error(f"Error fetching Stock #1 ({t1}): {err1}")
            elif err2:
                st.error(f"Error fetching Stock #2 ({t2}): {err2}")
            else:
                st.subheader(f"⚔️ Head-to-Head Comparison: {t1} ({d1}) vs. {t2} ({d2})")
                st.markdown(f"Comparing **{t1}** (matched against `{', '.join(res1['twin_tickers'])}`) vs **{t2}** (matched against `{', '.join(res2['twin_tickers'])}`).")

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

                # Plot Cumulative Returns
                ax1.plot(res1["rel_days"], res1["target_norm"].values, label=f"{t1} ({d1})", color="navy", lw=2.5)
                ax1.plot(res2["rel_days"], res2["target_norm"].values, label=f"{t2} ({d2})", color="crimson", lw=2.5)
                ax1.plot(res1["rel_days"], res1["twin_norm"].values, label=f"{t1} Twins", color="navy", linestyle="--", alpha=0.5)
                ax1.plot(res2["rel_days"], res2["twin_norm"].values, label=f"{t2} Twins", color="crimson", linestyle="--", alpha=0.5)
                ax1.axvline(0, color="black", linestyle=":", label="Inclusion (t=0)")
                ax1.set_title(f"Cumulative Return Trajectories: {t1} vs. {t2}")
                ax1.set_ylabel("Cumulative Return (%)")
                ax1.legend()

                # Plot Green Scores
                ax2.plot(res1["rel_days"], res1["green_score"].values, label=f"{t1} Green Score", color="navy", lw=2.5)
                ax2.plot(res2["rel_days"], res2["green_score"].values, label=f"{t2} Green Score", color="crimson", lw=2.5)
                ax2.axvline(0, color="black", linestyle=":")
                ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
                ax2.set_title("Empirical Green Score Comparison (CAR_target - CAR_twin)")
                ax2.set_xlabel("Trading Days Relative to Inclusion (t=0)")
                ax2.set_ylabel("Green Score (%)")
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)

                # Summary Comparison Matrix Table
                def get_val_at_t(rel_days, series, target_t):
                    idx = np.abs(rel_days - target_t).argmin()
                    return round(series.iloc[idx], 2)

                comp_data = []
                for label, r in [(t1, res1), (t2, res2)]:
                    rd = r["rel_days"]
                    gs = r["green_score"]
                    comp_data.append({
                        "Stock": label,
                        "Inclusion Date": r["event_date"],
                        "Peak Green Score (%)": round(gs.max(), 2),
                        "t=0 Score (%)": get_val_at_t(rd, gs, 0),
                        "+30d Score (%)": get_val_at_t(rd, gs, 30),
                        "+90d Score (%)": get_val_at_t(rd, gs, 90),
                        "+180d Score (%)": get_val_at_t(rd, gs, 180),
                        "Matched Twins": ", ".join(r["twin_tickers"])
                    })

                st.subheader("📊 Comparative Green Score Matrix Table")
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

# ==============================================================================
# MODE 3: CUSTOM CSV DATA INGESTION
# ==============================================================================
elif app_mode == "📁 Custom CSV Data Ingestion":
    st.header("📁 Custom Data Ingestion & Empirical Analysis")
    st.markdown("""
    Upload your proprietary or licensed dataset (e.g. CRSP mutual fund holdings, institutional constituent concentration weights, or custom stock return series).
    """)

    st.subheader("⚡ Quick-Load Research Project Benchmark Datasets")
    col1, col2, col3 = st.columns(3)
    with col1:
        load_cs = st.button("📊 Load Corporate Supply Dataset")
    with col2:
        load_c10 = st.button("📈 Load S&P 500 C10 & Active Share")
    with col3:
        load_event = st.button("🎯 Load Kroger (KR) Inclusion Event")

    preset_df = None
    if load_cs and os.path.exists("/workspace/out/corporate_supply_elasticity_multiasset.csv"):
        preset_df = pd.read_csv("/workspace/out/corporate_supply_elasticity_multiasset.csv")
    elif load_c10 and os.path.exists("/workspace/out/sp500_c10_active_share_historical.csv"):
        preset_df = pd.read_csv("/workspace/out/sp500_c10_active_share_historical.csv")
    elif load_event and os.path.exists("/workspace/out/index_inclusion_event_sample.csv"):
        preset_df = pd.read_csv("/workspace/out/index_inclusion_event_sample.csv")

    uploaded_file = st.file_uploader("Or Upload Your Own CSV File", type=["csv"])

    user_df = None
    if preset_df is not None:
        user_df = preset_df
        st.success("Loaded project benchmark dataset!")
    elif uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if user_df is not None:
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
    else:
        st.info("Please drag and drop a CSV file or click one of the preset loader buttons above.")

st.markdown("---")
st.caption("The Rational Coward Research Program — Dual-Mode Streamlit Web Application (v5.0)")
