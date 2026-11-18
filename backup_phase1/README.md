# Rajasthan Royals SupeRR Selector Season 2
## Advanced Cricket Analytics, Machine Learning & Pressure Performance Intelligence System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#)
[![Validation: Verified](https://img.shields.io/badge/Pipeline-Verified-emerald.svg)](#)

An autonomous, evidence-driven sports analytics intelligence platform engineered for the **Rajasthan Royals SupeRR Selector Season 2** competition. This system identifies cricketers who reliably elevate or sustain their performance in high-leverage match situations using historical ball-by-ball IPL (2022–2026) and T20 International data.

---

## 🏏 Official Top 10 Leaderboards (RR-PRI)

### Leaderboard A: Top 10 Batters Under Pressure

| Rank | Player Name | Pressure SR | Bayes-SR | Pressure Avg | Normal SR | SR Diff | Consistency | Archetype | RR-PRI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **#1** | **Ashutosh Sharma** | **197.5** | 177.5 | 33.6 | 159.2 | +38.3 | 0.569 | High-Impact Power Finisher | **81.00** |
| **#2** | **Rinku Singh** | **183.1** | 172.6 | 36.5 | 141.4 | +41.7 | 0.473 | High-Impact Power Finisher | **80.37** |
| **#3** | **Marcus Stoinis** | **203.4** | 182.3 | 19.7 | 161.3 | +42.1 | 0.532 | High-Volatility Accelerator | **78.94** |
| **#4** | **Ayush Badoni** | **188.9** | 172.2 | 34.0 | 134.3 | +54.6 | 0.529 | Adaptive Middle-Order Stabilizer | **78.92** |
| **#5** | **Tristan Stubbs** | **184.7** | 167.5 | 22.9 | 145.6 | +39.1 | 0.549 | High-Impact Power Finisher | **76.99** |
| **#6** | **Suryakumar Yadav** | **213.2** | 174.3 | 36.2 | 160.7 | +52.6 | 0.523 | Elite Dynamic Finisher | **75.27** |
| **#7** | **Tim David** | **188.2** | 167.0 | 38.4 | 166.5 | +21.7 | 0.544 | High-Impact Power Finisher | **73.88** |
| **#8** | **MS Dhoni** | **166.2** | 160.0 | 28.5 | 139.8 | +26.4 | 0.536 | Legendary Clutch Finisher | **72.35** |
| **#9** | **Liam Livingstone** | **191.7** | 172.5 | 26.3 | 158.5 | +33.2 | 0.485 | High-Volatility Accelerator | **72.16** |
| **#10** | **Jitesh Sharma** | **187.4** | 167.5 | 18.1 | 155.6 | +31.8 | 0.631 | Consistent Middle-Order Finisher | **71.81** |

---

### Leaderboard B: Top 10 Bowlers Under Pressure

| Rank | Player Name | Pressure Eco | Bayes-Eco | Pressure Wkts | Normal Eco | Eco Diff | Consistency | Archetype | RR-PRI |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **#1** | **Corbin Bosch** | **8.06** | 8.58 | 10 | 8.56 | -0.50 | 0.606 | Death-Over Restrictor | **85.51** |
| **#2** | **Dwayne Bravo** | **9.83** | 9.45 | 11 | 9.80 | +0.03 | 0.722 | Death-Over Specialist | **75.07** |
| **#3** | **Tim Southee** | **8.77** | 8.69 | 23 | 8.56 | +0.21 | 0.622 | Defensive Containment Specialist | **74.66** |
| **#4** | **Sunil Narine** | **8.13** | 8.32 | 12 | 7.02 | +1.11 | 0.704 | Elite Choke Spinner | **72.17** |
| **#5** | **Reece Topley** | **8.94** | 8.84 | 16 | 9.00 | -0.06 | 0.650 | Defensive Containment Specialist | **72.04** |
| **#6** | **Mark Wood** | **9.86** | 9.72 | 12 | 9.12 | +0.74 | 0.683 | Elite Strike Wicket-Taker | **71.49** |
| **#7** | **Jasprit Bumrah** | **8.23** | 8.42 | 23 | 7.46 | +0.77 | 0.631 | Generational Death Restrictor | **70.94** |
| **#8** | **Spencer Johnson** | **8.92** | 9.15 | 7 | 8.88 | +0.04 | 0.535 | High-Pace Enforcer | **68.99** |
| **#9** | **Mohsin Khan** | **8.78** | 8.85 | 12 | 8.56 | +0.22 | 0.629 | Left-Arm Restrictor | **68.93** |
| **#10** | **Ravichandran Ashwin** | **8.28** | 8.62 | 12 | 7.91 | +0.37 | 0.611 | Tactical Pressure Spinner | **68.57** |

---

## 🔬 Core Innovations

1. **Dynamic Ball-by-Ball Pressure Situation Engine:**
   - Evaluates match leverage immediately prior to every delivery (0–100 scale).
   - Incorporates Required Run Rate (RRR), Leverage Ratio (RRR/CRR), Wickets in Hand, Deliveries Remaining, and Target Proximity.
   - Separate models for 1st innings setting and 2nd innings chasing.
2. **Empirical Bayesian Shrinkage:**
   - Mitigates sample size bias by shrinking extreme estimates towards informed contextual priors.
   - Strike Rate: Prior = 130.0 with 50 pseudo-balls weight.
   - Economy: Prior = 9.00 with 60 pseudo-balls weight.
3. **Match Consistency & Stability:**
   - Incorporates coefficient of variation across individual matches ($C = \frac{1}{1 + \text{CV}}$).
   - Sensitivity testing across threshold and prior perturbations confirms **80.0% Top-10 ranking stability**.

---

## 📁 Repository Structure

```
RR-SuperRR-Selector-2027/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── bbb_with_pressure.parquet
│       ├── batting_features.parquet
│       └── bowling_features.parquet
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── pressure_engine.py
│   ├── feature_engineering.py
│   ├── bayesian_model.py
│   ├── player_ranking.py
│   ├── ml_models.py
│   ├── sensitivity_analysis.py
│   └── generate_pdf_report.py
│
├── outputs/
│   ├── reports/
│   │   ├── RR_SupeRR_Selector_Methodology_Report.pdf
│   │   ├── data_quality_report.json
│   │   ├── model_evaluation_report.json
│   │   ├── model_explainability_report.json
│   │   └── ranking_stability_report.json
│   ├── rankings/
│   │   ├── batters_final_ranked.csv
│   │   ├── bowlers_final_ranked.csv
│   │   └── RR_SuperRR_FINAL_Rankings.xlsx
│   └── charts/
│
├── dashboard/
│   ├── frontend/
│   │   └── index.html
│   └── backend/
│       └── main.py
│
├── tests/
│   └── test_pipeline.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Analysis Pipeline
```bash
python main.py
```

### 3. Run ML Modeling & Sensitivity Analysis
```bash
python src/ml_models.py
python src/sensitivity_analysis.py
```

### 4. Generate the Official PDF Report
```bash
python src/generate_pdf_report.py
```

### 5. Launch the Interactive Dashboard
```bash
python dashboard/backend/main.py
```
Open your browser and navigate to `http://127.0.0.1:8000` to interact with all 7 pages of the SupeRR Selector Pressure Intelligence Engine.

### 6. Run Automated Test Suite
```bash
python -m unittest discover tests
```

---

## 🏆 Key Findings for Rajasthan Royals IPL 2027 Auction

- **Lower-Order Finishing (No. 6/7):** Ashutosh Sharma (Rank #1) and Rinku Singh (Rank #2) demonstrate the highest boundary impact per pressure delivery across the entire Indian domestic pool.
- **Death Bowling Enforcement:** Corbin Bosch (Rank #1) offers elite run containment (8.06 economy) and yorker execution at high leverage.
- **Reliability over Reputation:** The framework identifies that several high-profile players experience negative pressure differentials, protecting franchise auction budget from overvaluing reputation.

