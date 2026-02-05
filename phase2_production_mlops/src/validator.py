import pandas as pd
import numpy as np

class FinFlowValidator:
    """
    Ensures that incoming raw data batches meet the schema requirements 
    before entering the model preprocessing pipeline.
    """
    
    def __init__(self, required_features):
        """
        Args:
            required_features (list): List of raw feature names expected.
        """
        self.required_features = required_features

    def validate_schema(self, df):
        """Checks if mandatory raw columns are present."""
        # Note: We check for raw features (e.g., 'industry'), not OHE features (e.g., 'industry_retail').
        missing_cols = [col for col in self.required_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Schema Error: Missing required features: {missing_cols}")
        return True

    def check_nulls(self, df):
        """Identifies null values in critical raw features."""
        # Only check columns that actually exist in the dataframe to avoid errors.
        available_features = [f for f in self.required_features if f in df.columns]
        null_report = df[available_features].isnull().sum()
        total_nulls = null_report.sum()
        
        if total_nulls > 0:
            print(f"⚠️ Warning: Found {total_nulls} null values in core features.")
        return total_nulls == 0

    def validate_data_types(self, df):
        """Ensures numeric logic is consistent for key drivers."""
        key_numeric = ['loan_to_income', 'annual_income', 'loan_amount']
        for col in key_numeric:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                raise TypeError(f"❌ Data Type Error: Feature '{col}' must be numeric.")
        return True

    def run_all_checks(self, df):
        """Executes the full validation suite."""
        print("🔍 Starting Data Validation Pipeline...")
        try:
            self.validate_schema(df)
            self.validate_data_types(df)
            self.check_nulls(df)
            print("✅ Data Validation Passed.")
            return True
        except Exception as e:
            # We print the error message specifically to help debugging.
            print(f"🛑 Data Validation Failed: {e}")
            return False