import sys
import os
import logging
from datetime import datetime

# ==============================================================================
# 1. ROBUST PATH CONFIGURATION
# We use __file__ to locate the script's directory regardless of where it is 
# executed from, preventing 'ModuleNotFoundError'.
# ==============================================================================

# Absolute path to 'phase2_production_mlops'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the 'src' and 'logs' directories
SRC_PATH = os.path.join(BASE_PATH, 'src')
LOG_DIR = os.path.join(BASE_PATH, 'logs')

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Add 'src' to the system path so Python can find our custom modules
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# Now we can safely import our modular components 
try:
    from data_loader import load_config
    from inference import run_inference
    from monitor import run_monitoring_suite
except ImportError as e:
    print(f"🛑 Critical Import Error: {e}")
    sys.exit(1)

# ==============================================================================
# 2. AUDIT LOGGING SETUP
# Standardizing logs for regulatory compliance and system health tracking.
# ==============================================================================
LOG_FILE = os.path.join(LOG_DIR, 'system_audit.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'), 
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """
    Main execution loop for the FinFlow weekly credit risk assessment.
    Flow: Monitoring (Drift) -> Data Loading -> Validation -> Inference.
    """
    start_time = datetime.now()
    logging.info("🚀 --- Starting FinFlow Weekly Risk Pipeline ---")
    
    try:
        # STEP 1: MONITORING & DRIFT DETECTION
        # Checks if current data distribution (PSI) is still aligned with 
        # the training baseline to protect model precision.
        logging.info("Step 1: Running Data Drift Monitoring (PSI)...")
        run_monitoring_suite()
        
        # STEP 2: BATCH INFERENCE & DECISION TIERING
        # Loads the champion model (0.81 Precision) and applies the 0.29 
        # optimal profit threshold to generate risk tiers.
        logging.info("Step 2: Executing Batch Inference & Risk Tiering...")
        run_inference()
        
        duration = datetime.now() - start_time
        logging.info(f"✅ --- Pipeline Execution Successful (Duration: {duration}) ---")
        
    except Exception as e:
        # CRITICAL FAILURE HANDLING
        # Logs the specific error for debugging and exits with failure code.
        logging.error(f"🛑 --- Pipeline CRASHED: {str(e)} ---")
        sys.exit(1)

if __name__ == "__main__":
    main()