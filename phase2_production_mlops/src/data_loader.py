import yaml
import pandas as pd
import os

def load_config(config_path):
    """
    Loads the production YAML configuration file.
    Args:
        config_path (str): Path to the .yaml file.
    Returns:
        dict: Parsed configuration parameters.
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def get_latest_data(config):
    """
    Loads the weekly batch of new loan applications from the data storage.
    Args:
        config (dict): Configuration dictionary containing file paths.
    Returns:
        pd.DataFrame: The raw data for inference.
    """
    # Construct the path for the new weekly batch
    # In a real scenario, this could include a date string like 'new_batch_20260204.csv'
    data_path = os.path.join(config['paths']['data_storage'], 'loan_data.csv') 
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Error: Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded {len(df)} new applications for batch processing.")
    return df

# --- Local Test Logic ---
# This part only runs if you execute this file directly in VS Code
if __name__ == "__main__":
    # Point to the config file we just created
    CONFIG_PATH = r"C:\dev\quant_project\homework\phase2_production_mlops\config\prod_config.yaml"
    
    try:
        cfg = load_config(CONFIG_PATH)
        print("✅ Config File Loaded Successfully.")
        print(f"Current Decision Threshold: {cfg['business_logic']['decision_threshold']}")
        
        # Test data loading
        data = get_latest_data(cfg)
    except Exception as e:
        print(f"❌ An error occurred: {e}")