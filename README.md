# FinFlow: End-to-End Credit Risk Management System
**An Industrial-Grade R&D to MLOps Framework**

---

## Business Context & Project Goals
FinFlow was developed to modernize the small business lending process by transitioning from manual, subjective credit reviews to an automated, data-driven system.

* **Maximize Net Profitability**: Identify the mathematical "Sweet Spot" between interest income and default losses.
* **Operational Velocity**: Automate routine approvals to reduce processing time from days to seconds.
* **Risk Resilience**: Implement real-time monitoring to detect economic shifts before they impact the portfolio.

---

## Project Highlights (Business KPIs)
* **Model Precision**: **81.0%** — Accurately identifying high-risk defaulters to protect company capital.
* **Optimal Threshold**: **0.29** — Derived from profit-maximization curves to balance recovery vs. yield.
* **Automation Rate**: **~70%** — Significant reduction in manual workload; credit officers focus only on "Medium Risk" cases.
* **System Stability**: Built-in **PSI (Population Stability Index)** monitoring to preemptively catch data drift.



---

## Data Methodology & Simulation
To ensure the system is robust enough for real-world application, the dataset was engineered to mirror the challenges of a high-quality credit portfolio.

### Controlled Class Imbalance
The target variable (`default`) was generated using a logic-based risk scoring function with a strategic intercept
$$RiskScore = \sum (\beta_i \cdot Feature_i) - 3.5 + \epsilon$$

* **Strategic Offset (-3.5)**: A negative offset was applied to lower the overall default probability.
* **Resulting Distribution**: This produced a **15% default rate**, creating a natural class imbalance that reflects a real-world "stable" lending environment.
* **Metric Selection**: This imbalance justified the use of **Precision** and **F1-Score** as primary metrics rather than Accuracy, ensuring the model remains sensitive to rare but high-impact default events.

---

## System Architecture
The project follows a rigorous lifecycle, transitioning from experimental research to a modular production environment.

### Phase 1: Research & Development (Jupyter Notebooks)
* **`01_Data_Quality.ipynb`**: EDA and cleaning of 5,000+ historical credit records.
* **`02_Feature_Engineering.ipynb`**: Construction of high-value ratios like `loan_to_income`.
* **`03_Model_Development.ipynb`**: Random Forest training, 5-Fold Cross-Validation, and Pipeline serialization.
* **`04_Business_Strategy.ipynb`**: Profit curve optimization and Model Explainability (SHAP).

### Phase 2: Production Pipeline (MLOps Framework)
* **`config/prod_config.yaml`**: Centralized management of paths and business thresholds.
* **`src/validator.py`**: The **Quality Gate** ensuring 100% schema integrity.
* **`src/inference.py`**: Integration of feature engineering and batch scoring.
* **`src/monitor.py`**: Calculation of **PSI** to prevent model decay.

---

## Technical Core: PSI Drift Monitoring
To adapt to changing market conditions, the system monitors **Data Drift** using the Population Stability Index.

* **PSI < 0.1**: Stable performance; system continues automated execution.
* **PSI > 0.2**: Significant drift; system triggers an alert for model retraining.

---

## Quick Start (Production Run)

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Execute Weekly Batch Run
Place the new data batch in `data_storage/loan_data.csv` and run:
```bash
python weekly_run.py
```

### 3. Audit & Verification

Open `phase1_model_development/05_Production_Verification.ipynb` to review the generated Risk Distribution and Potential ROI Report.

```text
quant_project/
├── data_storage/              # Raw data and training baselines
├── model_storage/             # Serialized model artifacts (.joblib)
├── phase1_model_development/  # R&D and Audit records
└── phase2_production_mlops/   # Production-ready MLOps pipeline
    ├── config/                # YAML configuration files
    ├── logs/                  # Audit logs and weekly decision reports (CSV)
    ├── src/                   # Modular Python source code
    └── weekly_run.py          # Master entry point