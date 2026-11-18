# The Rational Coward — Market Structure Lab & Research Program

An open-source simulation and empirical research toolkit exploring **how benchmark pressures, passive indexing, and agency constraints alter active risk-taking and asset price discovery**.

Based on **The Rational Coward Research Program** & **Further Research Addendum** (Nov 2026).

---

## 📌 Executive Summary

Modern financial markets have experienced a structural shift toward benchmark-linked and passive investing. This repository provides computational tools to analyze, test, and falsify two core theoretical propositions:

1. **Incumbent Risk Aversion & Entrant Polarization**: Established asset managers facing franchise-value loss rationally contract tracking error as passive flows increase. Conversely, prospective entrants facing high viability thresholds pursue volatile, attention-generating strategies, leading to risk polarization.
2. **Passive Flow Price Impact & Corporate Supply Inelasticity**: Mechanical benchmark demand drives valuation wedges ({target} - R_{twin}$) around index inclusion/deletion events (Green's Hypothesis). The magnitude of this distortion depends on informed arbitrage capacity and corporate share issuance/buyback elasticity (\gamma_{corp}).

---

## 🛠️ Repository Architecture & Deliverables



### 1.  — Dual-Mode Streamlit Web App
The interactive application features two top-level operating modes:

* **🧪 Theoretical Simulation Engine**:
  * **Module 1 (Agent-Based Simulation)**: Models 120 heterogeneous incumbent managers and 100 prospective entrants under benchmark flow pressure (\lambda_{passive}), corporate supply elasticity (\gamma_{corp}), expanded entrant volatility grids (up to 70%), and failure penalties.
  * **Module 2 (Active Share & Concentration Drift)**: Models mechanical active share erosion vs. top-10 benchmark concentration ({10}$).
  * **Module 3 (Multi-Asset Glidepath Extension)**: Models Target-Date Fund (TDF) cross-asset spillovers across Large-Cap Equities, Small-Cap Equities, and Fixed Income.
* **📊 Empirical Real-Data Laboratory**:
  * **Live  Integration**: Pulls live historical market data for historical S&P 500 inclusion events (**Tesla 2020**, **Palantir 2024**, **Super Micro 2024**, **Deckers Outdoor 2024**, or custom tickers).
  * **KNN Fundamental Twin Matcher**: Identifies candidate peer stocks matching pre-event risk profiles.
  * **Green Score Event Study**: Calculates Cumulative Abnormal Returns (CAR) 30$ days around inclusion dates to measure mechanical index demand impact.
* **📁 Custom CSV Data Ingestion**: Drag-and-drop interface for proprietary fund holdings or institutional constituent datasets.

### 2.  — Google Colab Notebook
A self-contained Python notebook designed for Google Colab, containing all core simulation classes, mathematical formulations, KNN twin algorithms, and plot generators.

---

## 🚀 Quickstart & Deployment

### Running Locally (Streamlit)

1. Clone this repository or download the files.
2. Install dependencies:
   
3. Launch the Streamlit dashboard:
   

### Running in Google Colab

1. Download  from the Studio panel.
2. Upload the file to [Google Colab](https://colab.research.google.com/).
3. Execute Cell 1 to initialize dependencies (, , , , ).

### Deploying to Streamlit Community Cloud

1. Create a public GitHub repository containing , , and .
2. Connect your repo to Streamlit Community Cloud.
3. Set  as the entry point and deploy.

---

## 🔬 Disconfirming Criteria & Falsification Standards

Per Section D of the Addendum, this research program provides explicit disconfirming standards. Key findings that would falsify or bound the theoretical model include:

1. **Active Share Decomposition Null**: Absence of a material manager-choice component after controlling for mechanical {10}$ concentration drift.
2. **Distribution Exposure Gradient Failure**: Lack of differential tracking error contraction across asset managers grouped by enterprise/plan-distribution exposure.
3. **Corporate & Arbitrage Neutralization**: Empirical evidence showing corporate share supply (\gamma_{corp}) and informed arbitrage entry respond sufficiently to neutralize persistent valuation wedges post-index inclusion.

---

## 📚 Selected References

* **Cremers & Petajisto (2009)** — *How Active Is Your Fund Manager?*, Review of Financial Studies.
* **Shleifer & Vishny (1997)** — *The Limits of Arbitrage*, Journal of Finance.
* **Grossman & Stiglitz (1980)** — *On the Impossibility of Informationally Efficient Markets*, American Economic Review.
* **Greenwood & Sammon (2025)** — *The Disappearing Index Effect*, Journal of Finance.
* **Coles, Heath, & Ringgenberg (2022)** — *On Index Investing*, Journal of Financial Economics.
