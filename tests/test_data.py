"""
Data Pipeline Tests - Health MLOps Project
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


class TestDataIngestion:
    """Test data collection and preprocessing"""

    def test_processed_data_exists(self):
        """Check if processed data files exist"""
        assert os.path.exists('data/processed/features_engineered.csv'), \
            "Features engineered file missing"
        assert os.path.exists('data/processed/centralized_data.pkl'), \
            "Centralized data file missing"

    def test_data_schema(self):
        """Verify data has correct schema"""
        if os.path.exists('data/processed/features_engineered.csv'):
            df = pd.read_csv('data/processed/features_engineered.csv')

            # Check for key columns
            required_cols = ['date', 'sick_percentage', 'health_risk_score']
            for col in required_cols:
                assert col in df.columns, f"Missing required column: {col}"

            # Check data types
            assert pd.api.types.is_numeric_dtype(df['sick_percentage']), \
                "sick_percentage should be numeric"

    def test_no_null_critical_features(self):
        """Check critical features have no null values"""
        if os.path.exists('data/processed/features_engineered.csv'):
            df = pd.read_csv('data/processed/features_engineered.csv')

            critical_cols = ['sick_percentage', 'health_risk_score']
            for col in critical_cols:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    assert null_count == 0, f"{col} has {null_count} null values"

    def test_data_ranges(self):
        """Check data values are within expected ranges"""
        if os.path.exists('data/processed/features_engineered.csv'):
            df = pd.read_csv('data/processed/features_engineered.csv')

            if 'sick_percentage' in df.columns:
                assert df['sick_percentage'].min() >= 0, "sick_percentage has negative values"
                assert df['sick_percentage'].max() <= 100, "sick_percentage exceeds 100%"


class TestDataDrift:
    """Test data drift detection"""

    def test_drift_detection_script_exists(self):
        """Check if drift detection script exists"""
        assert os.path.exists('src/data_drift_detection.py'), \
            "Data drift detection script missing"


class TestFeatureEngineering:
    """Test feature engineering pipeline"""

    def test_feature_engineering_script_exists(self):
        """Check if feature engineering script exists"""
        assert os.path.exists('src/feature_engineering.py'), \
            "Feature engineering script missing"

    def test_feature_names_file_exists(self):
        """Check if feature names are saved"""
        if os.path.exists('data/processed/feature_names.txt'):
            with open('data/processed/feature_names.txt', 'r') as f:
                features = f.read().strip().split('\n')
                assert len(features) > 0, "No features found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
