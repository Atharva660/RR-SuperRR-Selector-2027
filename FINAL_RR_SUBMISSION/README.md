# Rajasthan Royals SupeRR Selector Season 2
## Advanced Cricket Analytics, Machine Learning & Pressure Performance Intelligence System
### Independent Model Audit, Cricket Law Verification & IPL 2027 Auction Roster Intelligence

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)
[![Validation: Verified](https://img.shields.io/badge/Pipeline-Verified-emerald.svg)](#)
[![Tests: 16/16 Passed](https://img.shields.io/badge/Pytest-16%2F16%20Passed%20(100%25)-brightgreen.svg)](#)
[![Reconciliation: 100%](https://img.shields.io/badge/Data%20Reconciliation-100%25%20(0%20Discrepancy)-success.svg)](#)

An autonomous, mathematically defensible, and sports-science-validated cricket analytics platform engineered for the **Rajasthan Royals SupeRR Selector Season 2** competition. This platform identifies cricketers who consistently elevate or maintain their effectiveness in high-leverage match situations, providing unassailable player recommendations for Rajasthan Royals' IPL 2027 Mega-Auction strategy.

---

## 🏏 Official Category A Leaderboards (Primary IPL Qualified)

Eligibility criteria for Category A:
- **Batters**: Minimum 50 combined pressure balls, $\ge$ 25 IPL pressure balls, $\ge$ 10 total matches.
- **Bowlers**: Minimum 60 combined pressure legal balls, $\ge$ 30 IPL pressure legal balls, $\ge$ 10 total matches.

### Leaderboard A: Top 10 Batters Under Pressure

| Rank | Player Name | P-SR | Bayes-SR | 90% SR CI | P-Avg | IPL Pressure Balls | SR Diff | Consistency | Top-10 Prob | RR-PRI | Tactical Archetype |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | **Rinku Singh** | **198.5** | **180.1** | [166.5, 193.7] | **66.8** | 193 | +46.0 | 0.559 | **94.7%** | **80.29** | Elite Death-Over Finisher |
| **#2** | **David Miller** | **190.2** | **179.1** | [165.8, 192.5] | **40.1** | 162 | +40.9 | 0.504 | **97.3%** | **70.51** | Proven Middle-Order Finisher |
| **#3** | **MS Dhoni** | **170.7** | **160.2** | [147.5, 172.9] | **46.6** | 191 | +21.0 | 0.691 | **62.0%** | **68.50** | Legendary Clutch Finisher |
| **#4** | **Tristan Stubbs** | **181.6** | **169.6** | [156.2, 182.9] | **42.5** | 152 | +32.1 | 0.521 | **74.7%** | **67.49** | High-Impact Power Accelerator |
| **#5** | **Tilak Varma** | **179.0** | **168.7** | [155.6, 181.9] | **38.7** | 141 | +36.0 | 0.584 | **82.0%** | **66.73** | Dynamic Anchor & Finisher |
| **#6** | **Tim David** | **193.5** | **176.2** | [162.6, 189.9] | **34.2** | 159 | +15.8 | 0.561 | **70.7%** | **64.07** | High-Impact Power Finisher |
| **#7** | **Ashutosh Sharma** | **204.7** | **177.2** | [162.3, 192.0] | **29.3** | 123 | +37.9 | 0.544 | **62.0%** | **62.95** | High-Volatility Power Hitter |
| **#8** | **Dhruv Jurel** | **189.8** | **171.9** | [157.8, 186.1] | **27.9** | 143 | +38.6 | 0.527 | **63.3%** | **62.03** | Middle-Order Crisis Stabilizer |
| **#9** | **Jos Buttler** | **195.6** | **163.6** | [148.6, 178.6] | **44.3** | 59 | +40.2 | 0.557 | **28.7%** | **61.44** | Top-Order Explosive Match-Winner |
| **#10** | **Naman Dhir** | **206.7** | **163.5** | [148.5, 178.5] | **36.8** | 47 | +38.3 | 0.531 | **31.3%** | **60.05** | Emerging Impact Finisher |

---

### Leaderboard B: Top 10 Bowlers Under Pressure

| Rank | Player Name | P-Eco | Bayes-Eco | 90% Eco CI | P-Wkts | IPL Legal Balls | Eco Diff | Consistency | Top-10 Prob | RR-PRI | Tactical Archetype |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | **Corbin Bosch** | **8.70** | **8.83** | [8.12, 9.54] | 15 | 38 | -0.07 | 0.758 | 15.3% | **59.41** | Death Containment Seamer *(Cat B Candidate)* |
| **#2** | **Sunil Narine** | **6.82** | **7.19** | [6.70, 7.68] | **20** | **412** | +0.05 | 0.648 | **96.7%** | **59.29** | Elite Choke Spinner |
| **#3** | **Karn Sharma** | **8.43** | **8.65** | [7.96, 9.34] | 7 | 69 | +1.57 | 0.703 | 0.7% | **56.90** | Attacking Leg-Spinner |
| **#4** | **Dwayne Bravo** | **8.70** | **8.76** | [8.05, 9.47] | 10 | 91 | +0.01 | 0.699 | 0.0% | **55.14** | Legendary Death Specialist *(Retired)* |
| **#5** | **Eshan Malinga** | **8.89** | **9.00** | [8.37, 9.63] | **26** | **207** | +0.17 | 0.665 | **70.7%** | **54.87** | Sling-Action Yorker Specialist |
| **#6** | **Riyan Parag** | **7.78** | **8.56** | [7.90, 9.22] | 2 | 45 | +0.92 | 0.822 | 0.0% | **54.86** | Defensive Part-Time Spinner *(Cat B Candidate)* |
| **#7** | **Matheesha Pathirana**| **9.22** | **9.15** | [8.56, 9.74] | **42** | **305** | -0.39 | 0.706 | **57.3%** | **54.52** | Generational Death Restrictor |
| **#8** | **Mitchell Santner**| **7.71** | **8.17** | [7.56, 8.78] | **20** | **159** | -0.27 | 0.660 | **66.0%** | **54.32** | Precision Defensive Spinner |
| **#9** | **Noor Ahmad** | **8.35** | **8.47** | [7.91, 9.03] | **26** | **331** | -0.34 | 0.681 | **68.7%** | **54.05** | Attacking Left-Arm Wrist Spinner |
| **#10**| **Wanindu Hasaranga**| **8.42** | **8.55** | [7.96, 9.14] | **38** | **149** | -0.48 | 0.635 | **84.0%** | **53.35** | High-Strike Wicket Spinner |

> [!NOTE]
> **Audit Reclassification Notice**: Corbin Bosch and Riyan Parag, while technically exceeding the initial $\ge 30$ ball cut-off, exhibited low bootstrap top-10 probabilities (15.3% and 0.0% respectively) due to small IPL sample sizes. For auction bidding, front-line IPL targets are **Sunil Narine**, **Matheesha Pathirana**, **Eshan Malinga**, **Mitchell Santner**, **Noor Ahmad**, **Wanindu Hasaranga**, and **Jasprit Bumrah** (IPL Balls: 434, Wkts: 20, Prob: 51.3%).

---

## 🔬 Core Innovations & Methodological Rigour

### 1. 100% Delivery-Level Reconciliation (0 Discrepancy)
The preliminary report noted 194,014 deliveries against 200,999 raw rows. Our independent audit forensic analysis fully classified all 6,985 unmapped deliveries:
- **IPL (2022–2026)**: 367 played matches, 87,687 deliveries (43.63%)
- **Men's Emerging Teams Asia Cup (2024)**: 15 played matches, 3,519 deliveries (1.75%)
- **Asia Cup Rising Stars (2025)**: 15 played matches, 3,466 deliveries (1.72%)
- **T20 International (T20I)**: 486 played matches, 106,327 deliveries (52.90%)
- **Abandoned Matches**: Exactly 16 scheduled matches had 0 balls bowled due to weather washouts.
- **Total**: $87,687 + 6,985 + 106,327 = \mathbf{200,999}$ deliveries across $\mathbf{883}$ scheduled fixtures. **Discrepancy = 0**.

### 2. Strict MCC Cricket Law Compliance
- **Balls Faced (MCC Law 22)**: 7,241 wides were correctly excluded from striker balls faced. No-balls count as balls faced.
- **Bowler Legal Balls (MCC Laws 21 & 22)**: Deliveries with wides or overstep no-balls are excluded from legal overs.
- **Bowler Runs Conceded (MCC Law 21.18)**: 4,603 runs from byes, leg-byes, and penalties are strictly zeroed out from bowler economy.
- **Bowler Wickets (MCC Law 38)**: Bowler credit is strictly restricted to Bowled, Caught, LBW, Stumped, and Hit Wicket. Exactly 600 run-outs and 2 retire-outs were removed from bowler wicket tallies.

### 3. Zero-Leakage Dynamic Pressure Engine
Evaluates match leverage immediately prior to every delivery on a continuous 0–100 scale:
- Computed via vectorised pre-state subtraction: $\text{Cumulative State}_{t-1} = \text{cumsum}(x)_t - x_t$, guaranteeing zero future-ball information contamination.
- Chases model non-linear urgency using Required Run Rate (RRR), leverage ratio (RRR / CRR), wickets in hand, and balls remaining.
- First-innings models compute phase urgency and par score deficit against empirical IPL phase par rates.

### 4. Empirical Bayesian Shrinkage & Credible Intervals
Conjugate empirical Bayes models shrink small-sample noise towards tournament context priors:
- **Batting Strike Rate**: Normal-Normal conjugate shrinkage ($\mu_0 = 135.0$, $N_0 = 60$ balls) with analytical 90% credible intervals.
- **Batting Average**: Hazard rate shrinkage ($\text{Avg}_0 = 25.0$, $D_0 = 4$ dismissals), eliminating division-by-zero infinities.
- **Bowling Economy**: Rate shrinkage ($\text{Eco}_0 = 8.80$ RPO, $N_0 = 72$ legal balls).
- **Sample Exposure Reliability**: Exponential saturation $S = 1 - e^{-n / 120}$, ensuring 200+ ball veterans receive full statistical credit.

### 5. Match-Cluster Bootstrap Resampling (150 Iterations)
Recognising that deliveries within the same match share correlated pitch, dew, and weather conditions, we executed a 150-iteration match-cluster bootstrap. Entire matches were resampled with replacement, yielding empirical top-10 probabilities and rank confidence intervals.

### 6. Machine Learning Temporal Benchmarking
Delivery-level boundary hit probability under pressure was benchmarked using a match-grouped temporal split (Train: 2022–2024, Test: 2025–2026):
- **Naive Frequency Baseline**: ROC-AUC 0.5000, Brier Score 0.1498
- **Logistic Regression**: ROC-AUC 0.5829, PR-AUC 0.2303, Brier Score 0.1478
- **XGBoost**: ROC-AUC 0.5728, PR-AUC 0.2202, Brier Score 0.1487
- Outperformed baseline with superior probability calibration, functioning as informational priors.

---

## 📁 Repository Structure

```
RR-SuperRR-Selector-2027/
│
├── FINAL_RR_SUBMISSION/                 # Official competition submission directory
│   ├── final_top10_batters.csv          # Audited Top 10 Batters Leaderboard
│   ├── final_top10_bowlers.csv          # Audited Top 10 Bowlers Leaderboard
│   ├── final_rankings.xlsx              # 6-tab comprehensive workbook
│   ├── final_methodology_report.pdf     # Master 22-section research report
│   ├── model_audit_report.pdf           # Independent codebase audit & corrections
│   ├── model_validation_report.pdf      # ML temporal benchmarks & calibration
│   ├── ranking_stability_report.pdf     # Bootstrap resampling & sensitivity
│   ├── data_quality_report.pdf          # 100% reconciliation & MCC compliance
│   ├── test_results.txt                 # Complete pytest execution log (100% pass)
│   ├── test_results.xml                 # JUnit XML for automated evaluation
│   └── README.md                        # Submission documentation
│
├── data/
│   ├── raw/                             # Original raw datasets
│   └── processed/                       # Cleaned parquet master files
│
├── src/                                 # Audited analytics source code
│   ├── data_loader.py                   # Multi-dataset loading & mapping
│   ├── data_cleaning.py                 # MCC cricket laws & reconciliation
│   ├── pressure_engine.py               # Zero-leakage dynamic pressure engine
│   ├── feature_engineering.py           # Competition weighting & metrics
│   ├── bayesian_model.py                # Empirical Bayes shrinkage & CIs
│   ├── player_ranking.py                # Audited RR-PRI & tier classification
│   ├── bootstrap_validation.py          # Match-cluster bootstrap resampling
│   ├── ml_models.py                     # Temporal split ML benchmark
│   ├── sensitivity_analysis.py          # Parametric stability tests
│   └── generate_pdf_report.py           # Publication PDF report suite
│
├── tests/
│   └── test_cricket_intelligence.py     # 16-test comprehensive automated suite
│
├── outputs/
│   ├── reports/                         # PDF & JSON reports
│   ├── rankings/                        # CSV & Excel tables
│   └── charts/                          # High-resolution visualisations
│
├── dashboard/
│   ├── frontend/index.html              # Synchronized interactive intelligence UI
│   └── backend/main.py                  # FastAPI analytics backend
│
├── main.py                              # Master automated pipeline script
└── requirements.txt                     # Pinned dependencies
```

---

## 🚀 Execution & Verification

### Run Complete Analytics Pipeline
```bash
python main.py
```

### Run Automated Pytest Test Suite
```bash
pytest -v tests/test_cricket_intelligence.py
```
*Current Status: 16 Passed, 0 Failed (100% Pass Rate).*

### Launch Interactive Intelligence Dashboard
```bash
cd dashboard/backend
uvicorn main:app --reload --port 8000
```
Open `dashboard/frontend/index.html` in your browser.

---

## 🏆 Final Rajasthan Royals Auction Strategy Summary
- **Primary Finisher Target**: **Rinku Singh** (RR-PRI 80.29, 94.7% Top-10 stability, 198.5 SR under pressure).
- **Anchor & Finisher Target**: **Tilak Varma** (RR-PRI 66.73, 82.0% Top-10 stability, 179.0 SR under pressure).
- **Benchmark Spinner**: **Sunil Narine** (RR-PRI 59.29, 96.7% Top-10 stability, 6.82 Economy under pressure).
- **Benchmark Strike Bowler**: **Matheesha Pathirana** (42 pressure wickets, premier death overs specialist).
- **Left-Arm Wrist Spinner**: **Noor Ahmad** (RR-PRI 54.05, 68.7% Top-10 stability, 331 IPL pressure balls).
