import joblib
import pandas as pd
import os
import datetime
import logging
from data_loader import load_config, get_latest_data
from validator import FinFlowValidator

# ==============================================================================
# BATCH INFERENCE ENGINE
# ==============================================================================

def run_inference():
    """
    Loads the production model, processes raw weekly data, and generates 
    risk-tiered lending decisions based on the 0.29 profit threshold.
    """
    
    # 1. INITIALIZATION & CONFIG 
    # We use absolute paths to ensure reliability across different environments.
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CONFIG_PATH = os.path.join(BASE_PATH, 'config', 'prod_config.yaml')
    cfg = load_config(CONFIG_PATH)
    
    # 2. LOAD MODEL ARTIFACTS
    logging.info("🤖 Loading champion model artifacts from storage...")
    artifacts = joblib.load(cfg['paths']['model_path'])
    
    # The 'model' is a full Scikit-Learn Pipeline (Preprocessors + Random Forest).
    model = artifacts['model']
    
    # We validate RAW features because the Pipeline handles OHE internally.
    required_raw_features = artifacts['numeric_features'] + artifacts['categorical_features']
    
    # 3. FETCH DATA
    raw_data = get_latest_data(cfg)
    
    # 4. FEATURE ENGINEERING LAYER 
    # Replicating the logic from Notebook 02 for the raw incoming batch.
    logging.info("🛠️ Applying feature engineering to raw batch...")
    
    # Calculate Ratio Metrics
    if 'loan_to_income' not in raw_data.columns:
        raw_data['loan_to_income'] = raw_data['loan_amount'] / raw_data['annual_income']
        
    if 'rev_to_loan_ratio' not in raw_data.columns:
        raw_data['rev_to_loan_ratio'] = (raw_data['monthly_revenue'] * 12) / raw_data['loan_amount']
        
    if 'rev_per_employee' not in raw_data.columns:
        raw_data['rev_per_employee'] = (raw_data['monthly_revenue'] * 12) / raw_data['num_employees']

    # Map Credit Tiers
    if 'credit_tier' not in raw_data.columns:
        def get_credit_tier(score):
            if score >= 750: return 'Elite'
            elif score >= 650: return 'Average'
            else: return 'Subprime'
        raw_data['credit_tier'] = raw_data['credit_score'].apply(get_credit_tier)

    # 5. DATA VALIDATION (QUALITY GATE) 
    # Ensure all required raw columns are present before prediction.
    validator = FinFlowValidator(required_raw_features)
    if not validator.run_all_checks(raw_data):
        logging.error("🛑 Inference aborted: Input data does not match model schema.")
        return
    
    # 6. PROBABILITY PREDICTION
    logging.info("🔮 Generating default probability scores...")
    # Extract probability for Class 1 (Default)
    probs = model.predict_proba(raw_data)[:, 1]
    
    # 7. BUSINESS DECISION LOGIC
    # Applying the 0.29 threshold optimized for FinFlow capital recovery.
    threshold = cfg['business_logic']['decision_threshold']
    results = raw_data.copy()
    results['default_probability'] = probs
    
    def segment_risk(p):
        if p < 0.2: 
            return '🟢 Low Risk (Auto-Approve)'
        elif p < threshold: 
            return '🟡 Medium Risk (Manual Review)'
        else: 
            return '🔴 High Risk (Auto-Reject)'
            
    results['decision_tier'] = results['default_probability'].apply(segment_risk)
    
    # 8. OUTPUT PERSISTENCE
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"weekly_decision_report_{timestamp}.csv"
    output_path = os.path.join(cfg['paths']['output_logs'], output_filename)
    
    results.to_csv(output_path, index=False)
    logging.info(f"✅ Production Report Generated: {output_path}")

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    run_inference()