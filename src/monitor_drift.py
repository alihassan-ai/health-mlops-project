"""
Model Drift Monitoring - Health MLOps Project
Monitors for model performance degradation and data distribution changes
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
from scipy.stats import ks_2samp


def monitor_drift():
    """Monitor for model and data drift"""

    print("=" * 80)
    print("MODEL & DATA DRIFT MONITORING")
    print("=" * 80)

    drift_detected = False
    warnings = []

    # Check if drift detection results exist
    print("\n[1/2] Checking data drift...")
    drift_report_path = 'reports/drift/drift_report.csv'

    if os.path.exists(drift_report_path):
        df = pd.read_csv(drift_report_path)
        print(f"  ✓ Loaded drift report: {len(df)} features monitored")

        # Check for features with significant drift
        if 'p_value' in df.columns:
            significant_drift = df[df['p_value'] < 0.05]

            if len(significant_drift) > 0:
                drift_detected = True
                print(f"\n  ⚠️  Drift detected in {len(significant_drift)} features:")

                for idx, row in significant_drift.head(5).iterrows():
                    feature = row.get('feature', 'Unknown')
                    p_val = row.get('p_value', 0)
                    print(f"    - {feature}: p-value = {p_val:.4f}")

                warnings.append(
                    f"Data drift detected in {len(significant_drift)} features"
                )
            else:
                print("  ✅ No significant drift detected")
    else:
        print("  ⚠️  Drift report not found, running drift detection...")
        # Try to run drift detection
        if os.path.exists('src/data_drift_detection.py'):
            import subprocess
            try:
                subprocess.run(['python', 'src/data_drift_detection.py'],
                             check=True, capture_output=True)
                print("  ✓ Drift detection completed")
            except Exception as e:
                warnings.append(f"Failed to run drift detection: {str(e)}")

    # Check model performance drift
    print("\n[2/2] Checking model performance...")

    baseline_results_path = 'models/evaluation/baseline_results.csv'
    if os.path.exists(baseline_results_path):
        df = pd.read_csv(baseline_results_path)

        if 'r2_test' in df.columns:
            current_r2 = df['r2_test'].max()
            print(f"  Current best R²: {current_r2:.4f}")

            # Check if below warning threshold
            if current_r2 < 0.72:
                warnings.append(
                    f"Model R² ({current_r2:.4f}) approaching retraining threshold"
                )
                print(f"  ⚠️  Performance degradation warning")
            else:
                print(f"  ✅ Model performance stable")
    else:
        print("  ⚠️  Model results not found")

    # Summary
    print("\n" + "=" * 80)
    if drift_detected:
        print("⚠️  DRIFT DETECTED - Consider model retraining")
    else:
        print("✅ NO SIGNIFICANT DRIFT DETECTED")

    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")

    print("=" * 80)

    # Generate monitoring report
    report = {
        'timestamp': datetime.now().isoformat(),
        'drift_detected': drift_detected,
        'warnings': warnings
    }

    # Save report
    os.makedirs('reports', exist_ok=True)
    report_df = pd.DataFrame([report])
    report_path = 'reports/monitoring_latest.csv'
    report_df.to_csv(report_path, index=False)
    print(f"\n✓ Monitoring report saved: {report_path}")

    return not drift_detected


if __name__ == '__main__':
    import sys
    success = monitor_drift()

    # Exit with warning code if drift detected
    if not success:
        sys.exit(1)
