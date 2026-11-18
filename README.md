# The Rational Coward — Market Structure Lab & Simulation Suite

An open-source research and agent-based simulation suite examining how mechanical index demand, benchmark-linked flow pressure, and agency frictions impact market structure, active risk allocation, and patient fundamental price discovery.

Based on **"The Rational Coward Research Program"** and **"Further Research Addendum"** (Nov 2026), incorporating empirical frameworks from Green's hypothesis and the **Market Structure Lab**.

---

## 📌 Project Overview

As benchmark-linked assets grow and commercial penalties for benchmark deviation increase, established managers may rationally accept less active risk. Meanwhile, new entrants facing high minimum viable scale may pursue volatile, attention-generating strategies. 

This repository provides an end-to-end framework to simulate and empirically evaluate:
1. **Agent-Based Market Structure Dynamics:** 120 heterogeneous incumbent asset managers vs. 100 prospective entrants under flow pressure, informed arbitrage capacity limits, and corporate share supply responses.
2. **Green Score & Twin Matching:** K-Nearest Neighbors (KNN) fundamental twin matching around index inclusion/deletion event windows to isolate mechanical demand shocks from underlying business fundamentals.
3. **No-Trade Active Share Drift:** Isolating the mechanical erosion of Active Share driven by rising top-10 benchmark concentration.
4. **Multi-Asset TDF Glidepath Spillovers:** 60-period cross-asset spillover trajectories across Large-Cap Equities, Small-Cap Equities, and Fixed Income under PPA 2006 QDIA target-date fund default allocations.

---

## 📂 Repository Layout

```
.
├── app.py                                # Streamlit interactive web application
├── rational_coward_further_research-v3.ipynb  # Full research notebook (Google Colab ready)
├── requirements.txt                      # Dependencies for Streamlit & Colab
└── README.md                             # Project documentation & reproduction guide
```

---

## 🚀 Quickstart

### 1. Run Locally

```bash
# Clone repository and navigate to folder
cd market-structure-lab

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

### 2. Run in Google Colab

1. Open Google Colab.
2. Select Upload and upload `rational_coward_further_research-v3.ipynb`.
3. Execute Cell 1 to initialize the environment and run all four simulation modules sequentially.

### 3. Deploy to Streamlit Community Cloud

1. Push this repository to your code host.
2. Sign in to Streamlit Community Cloud.
3. Connect your repository and set `app.py` as the main entry point.

---

## 🔬 Core Simulation Modules

### Module 1: Agent-Based Market Simulation
- **Incumbent Optimization:** Models tracking error decisions of 120 incumbent asset managers protecting franchise values against benchmark pressure.
- **Entrant Grid Search:** Expands the entrant volatility risk grid up to 70% under performance-contingent capital-raising hurdles and failure penalties.
- **Corporate Supply Response:** Incorporates equity issuance and buyback elasticity to evaluate market elasticity.

### Module 2: Market Structure Lab (Green Score)
- **Twin Matcher:** Uses normalized KNN matching (K=3) across Market Cap, P/E Ratio, Revenue Growth, and Volatility.
- **Green Score Calculation:** Green Score = CAR_target - CAR_twin

### Module 3: Active Share Drift Counterfactual
- Simulates historical top-10 concentration trajectory.
- Disentangles mechanical passive holdings drift from active portfolio manager rebalancing choices.

### Module 4: Multi-Asset Glidepath Extension
- Simulates TDF glidepath allocations (55% / 15% / 30%).
- Models cross-asset spillover effects and price distortion wedges across Large-Cap, Small-Cap, and Bond markets.

---

## 🎯 Disconfirming & Falsification Standards

Per Section D of the Further Research Addendum, three key findings would disconfirm the core theoretical claims:
1. **No active component remaining** after controlling for mechanical top-10 concentration drift.
2. **Absence of an enterprise/distribution exposure gradient** among institutional managers.
3. **Sufficiently elastic informed capital and corporate share supply** to quickly extinguish persistent valuation wedges following benchmark demand shocks.

---

## 📚 References & Citation

- Cremers and Petajisto (2009), *How Active Is Your Fund Manager?*, Review of Financial Studies.
- Grossman and Stiglitz (1980), *On the Impossibility of Informationally Efficient Markets*, AER.
- Shleifer and Vishny (1997), *The Limits of Arbitrage*, Journal of Finance.
- Greenwood and Sammon (2025), *The Disappearing Index Effect*, Journal of Finance.
- Haddad, Huebner and Loualiche (2025), *How Competitive Is the Stock Market?*, AER.

---
*Open for replication, falsification, and extension.*
