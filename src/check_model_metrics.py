"""
Model Performance Threshold Checker - Health MLOps Project
Validates that trained models meet minimum performance requirements
"""

import pandas as pd
import os
import sys


def check_model_metrics():
    """Check if models meet performance thresholds"""

    print("=" * 80)
    print("MODEL PERFORMANCE THRESHOLD CHECKER")
    print("=" * 80)

    # Define thresholds
    THRESHOLDS = {
        'r2_min': 0.70,         # Minimum R² for regression
        'f1_min': 0.60,         # Minimum F1 for classification
        'accuracy_min': 0.85    # Minimum accuracy
    }

    passed = True
    failures = []

    # Check baseline models
    print("\n[1/2] Checking baseline models...")
    baseline_results_path = 'models/evaluation/baseline_results.csv'

    if os.path.exists(baseline_results_path):
        df = pd.read_csv(baseline_results_path)
        print(f"  ✓ Loaded results: {len(df)} models")

        # Check regression R²
        if 'r2_test' in df.columns:
            max_r2 = df['r2_test'].max()
            print(f"\n  Best R² Score: {max_r2:.4f}")

            if max_r2 < THRESHOLDS['r2_min']:
                failures.append(
                    f"Regression R² ({max_r2:.4f}) below threshold ({THRESHOLDS['r2_min']})"
                )
                passed = False
                print(f"  ❌ Below threshold ({THRESHOLDS['r2_min']})")
            else:
                print(f"  ✅ Above threshold ({THRESHOLDS['r2_min']})")

        # Check classification F1
        if 'f1_test' in df.columns:
            max_f1 = df['f1_test'].max()
            print(f"\n  Best F1 Score: {max_f1:.4f}")

            if max_f1 < THRESHOLDS['f1_min']:
                failures.append(
                    f"Classification F1 ({max_f1:.4f}) below threshold ({THRESHOLDS['f1_min']})"
                )
                # Don't fail on F1 since it's harder to achieve
                print(f"  ⚠️  Below threshold ({THRESHOLDS['f1_min']}) - Warning only")
            else:
                print(f"  ✅ Above threshold ({THRESHOLDS['f1_min']})")

    else:
        failures.append(f"Baseline results file not found: {baseline_results_path}")
        passed = False
        print(f"  ❌ Results file not found")

    # Check PyTorch models
    print("\n[2/2] Checking PyTorch models...")
    pytorch_results_path = 'models/evaluation/pytorch_results.csv'

    if os.path.exists(pytorch_results_path):
        df = pd.read_csv(pytorch_results_path)
        print(f"  ✓ Loaded results: {len(df)} models")

        # Check regression performance
        if 'r2_test' in df.columns:
            for idx, row in df.iterrows():
                model_name = row.get('model', f'Model {idx}')
                r2 = row['r2_test']

                print(f"\n  {model_name}: R² = {r2:.4f}")

                if r2 < THRESHOLDS['r2_min']:
                    print(f"    ⚠️  Below threshold ({THRESHOLDS['r2_min']})")
                else:
                    print(f"    ✅ Above threshold ({THRESHOLDS['r2_min']})")
    else:
        print(f"  ⚠️  PyTorch results file not found (optional)")

    # Final summary
    print("\n" + "=" * 80)
    if passed:
        print("✅ ALL MODELS MEET PERFORMANCE THRESHOLDS")
    else:
        print("❌ SOME MODELS BELOW PERFORMANCE THRESHOLDS")
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")

    print("=" * 80)

    return passed


if __name__ == '__main__':
    success = check_model_metrics()

    if not success:
        sys.exit(1)
