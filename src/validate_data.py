"""
Data Validation Script - Health MLOps Project
Validates data schema, quality, and integrity
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def validate_data():
    """Validate processed data for quality and completeness"""

    print("=" * 80)
    print("DATA VALIDATION - Health MLOps Project")
    print("=" * 80)

    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'passed': True,
        'errors': [],
        'warnings': []
    }

    # Check if data files exist
    print("\n[1/4] Checking data files existence...")
    required_files = [
        'data/processed/features_engineered.csv',
        'data/processed/centralized_data.pkl',
        'data/processed/merged_daily_data.csv'
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            validation_results['errors'].append(f"Missing file: {file_path}")
            validation_results['passed'] = False
            print(f"  ❌ {file_path}")
        else:
            print(f"  ✓ {file_path}")

    if not validation_results['passed']:
        print("\n❌ VALIDATION FAILED: Missing required data files")
        return validation_results

    # Load and validate features
    print("\n[2/4] Validating data schema...")
    df = pd.read_csv('data/processed/features_engineered.csv')

    required_columns = ['date', 'sick_percentage', 'health_risk_score']
    for col in required_columns:
        if col not in df.columns:
            validation_results['errors'].append(f"Missing required column: {col}")
            validation_results['passed'] = False
            print(f"  ❌ Missing column: {col}")
        else:
            print(f"  ✓ Column exists: {col}")

    # Check data quality
    print("\n[3/4] Checking data quality...")

    # Check for nulls in critical columns
    critical_cols = ['sick_percentage', 'health_risk_score']
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_results['warnings'].append(
                    f"{col} has {null_count} null values"
                )
                print(f"  ⚠️  {col}: {null_count} null values")
            else:
                print(f"  ✓ {col}: No nulls")

    # Check value ranges
    print("\n[4/4] Validating value ranges...")
    if 'sick_percentage' in df.columns:
        min_val = df['sick_percentage'].min()
        max_val = df['sick_percentage'].max()

        if min_val < 0 or max_val > 100:
            validation_results['errors'].append(
                f"sick_percentage out of range [0, 100]: [{min_val}, {max_val}]"
            )
            validation_results['passed'] = False
            print(f"  ❌ sick_percentage range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"  ✓ sick_percentage range: [{min_val:.2f}, {max_val:.2f}]")

    # Summary
    print("\n" + "=" * 80)
    if validation_results['passed']:
        print("✅ DATA VALIDATION PASSED")
    else:
        print("❌ DATA VALIDATION FAILED")
        print(f"\nErrors: {len(validation_results['errors'])}")
        for error in validation_results['errors']:
            print(f"  - {error}")

    if validation_results['warnings']:
        print(f"\nWarnings: {len(validation_results['warnings'])}")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")

    print("=" * 80)

    return validation_results


if __name__ == '__main__':
    result = validate_data()

    # Exit with error code if validation failed
    if not result['passed']:
        exit(1)
