import pandas as pd
import numpy as np
import os
import logging
from data_loader import load_config

# ==============================================================================
# 1. CORE STATISTICAL LOGIC: PSI CALCULATION
# ==============================================================================

def calculate_psi(expected, actual, buckets=10):
    """
    Calculates the Population Stability Index (PSI) to detect data drift.
    
    Args:
        expected (pd.Series): The baseline distribution (e.g., training data).
        actual (pd.Series): The current distribution (e.g., weekly batch).
        buckets (int): Number of bins for comparison. Default is 10.
    
    Returns:
        float: Calculated PSI value.
    """
    # Create bins based on the distribution of the 'Expected' data.
    breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
    
    # Calculate percentages for each bin
    expected_percents = pd.cut(expected, bins=breakpoints).value_counts(normalize=True).sort_index().values
    actual_percents = pd.cut(actual, bins=breakpoints).value_counts(normalize=True).sort_index().values
    
    # Use epsilon to avoid DivisionByZero or Log(0) errors.
    epsilon = 0.0001
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    
    # Apply the PSI formula
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

# ==============================================================================
# 2. MONITORING EXECUTION SUITE
# ==============================================================================

def run_monitoring_suite():
    """
    Executes drift detection on primary risk drivers.
    Includes an on-the-fly feature engineering layer to handle raw data batches.
    """
    # Load central configuration.
    CONFIG_PATH = r"C:\dev\quant_project\homework\phase2_production_mlops\config\prod_config.yaml"
    cfg = load_config(CONFIG_PATH)
    
    # Load datasets from specified storage paths.
    baseline_path = os.path.join(cfg['paths']['data_storage'], 'X_train.csv')
    current_path = os.path.join(cfg['paths']['data_storage'], 'loan_data.csv')
    
    train_baseline = pd.read_csv(baseline_path)
    current_batch = pd.read_csv(current_path)
    
    # PRIMARY FEATURE SELECTION
    # 'loan_to_income' was identified as the top driver in SHAP analysis.
    target_feature = 'loan_to_income'
    
    # FEATURE ENGINEERING LAYER
    # If the weekly batch is raw data, calculate the required feature manually.
    for df, name in [(train_baseline, "Baseline"), (current_batch, "Current Batch")]:
        if target_feature not in df.columns:
            logging.info(f"🛠️ '{target_feature}' missing in {name}. Calculating from raw columns...")
            if 'loan_amount' in df.columns and 'annual_income' in df.columns:
                df[target_feature] = df['loan_amount'] / df['annual_income']
            else:
                # Raise error if source columns for calculation are also missing.
                error_msg = f"Critical Error: Cannot calculate '{target_feature}' for {name}."
                logging.error(error_msg)
                raise KeyError(error_msg)

    # PSI CALCULATION
    logging.info(f"📡 Analyzing distribution stability for: '{target_feature}'...")
    psi_score = calculate_psi(train_baseline[target_feature], current_batch[target_feature])
    
    # ALERT LOGIC BASED ON PSI THRESHOLDS
    alert_limit = cfg['monitoring']['psi_alert_limit'] # Defined as 0.2
    
    logging.info(f"📊 Population Stability Index (PSI): {psi_score:.4f}")
    
    if psi_score > alert_limit:
        logging.warning(f"🚨 ALERT: Significant Data Drift Detected (PSI > {alert_limit})!")
    elif psi_score > 0.1:
        logging.info("⚠️ Warning: Moderate drift detected. Review incoming data quality.")
    else:
        logging.info("✅ Data distribution is stable. Proceeding to inference.")

if __name__ == "__main__":
    # Configure basic logging if run as a standalone script.
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run_monitoring_suite()