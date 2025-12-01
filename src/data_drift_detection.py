"""
Data Drift Detection System - Health MLOps Project
Monitors statistical changes in data distributions over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA DRIFT DETECTION SYSTEM")
print("Health MLOps Project")
print("=" * 80)

# Create output directories
os.makedirs('reports/drift', exist_ok=True)
os.makedirs('reports/drift/plots', exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading data...")
print("-" * 80)

# Load merged data
merged_data = pd.read_csv('data/processed/merged_daily_data.csv')
merged_data['date'] = pd.to_datetime(merged_data['date'])

# Sort by date
merged_data = merged_data.sort_values('date').reset_index(drop=True)

print(f"✓ Loaded {len(merged_data)} records")
print(f"  Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
print(f"  Number of features: {len(merged_data.columns)}")

# ============================================================================
# 2. SPLIT DATA INTO REFERENCE AND CURRENT WINDOWS
# ============================================================================

print("\n[STEP 2] Creating reference and current windows...")
print("-" * 80)

# Split data: First 60% as reference (training period), Last 40% as current
split_idx = int(len(merged_data) * 0.6)

reference_data = merged_data.iloc[:split_idx].copy()
current_data = merged_data.iloc[split_idx:].copy()

print(f"✓ Reference window (training period):")
print(f"  Records: {len(reference_data)}")
print(f"  Date range: {reference_data['date'].min()} to {reference_data['date'].max()}")

print(f"\n✓ Current window (monitoring period):")
print(f"  Records: {len(current_data)}")
print(f"  Date range: {current_data['date'].min()} to {current_data['date'].max()}")

# ============================================================================
# 3. STATISTICAL DRIFT DETECTION
# ============================================================================

print("\n[STEP 3] Detecting statistical drift...")
print("-" * 80)

# Select numerical features to monitor
numerical_features = [
    'avg_heart_rate', 'avg_spo2', 'avg_body_temp', 'avg_steps',
    'avg_pm25', 'avg_pm10', 'avg_no2', 'avg_aqi',
    'avg_temperature', 'avg_humidity', 'avg_pressure',
    'sick_percentage', 'health_risk_score'
]

# Filter to existing columns
numerical_features = [f for f in numerical_features if f in merged_data.columns]

drift_results = []

for feature in numerical_features:
    # Get reference and current distributions
    ref_values = reference_data[feature].dropna()
    curr_values = current_data[feature].dropna()
    
    if len(ref_values) == 0 or len(curr_values) == 0:
        continue
    
    # Kolmogorov-Smirnov Test
    # Tests if two samples come from the same distribution
    ks_statistic, ks_pvalue = ks_2samp(ref_values, curr_values)
    
    # Calculate distribution statistics
    ref_mean = ref_values.mean()
    curr_mean = curr_values.mean()
    mean_change = ((curr_mean - ref_mean) / ref_mean) * 100
    
    ref_std = ref_values.std()
    curr_std = curr_values.std()
    std_change = ((curr_std - ref_std) / ref_std) * 100
    
    # Determine drift severity
    # p-value < 0.05: Significant drift detected
    drift_detected = ks_pvalue < 0.05
    
    if ks_pvalue < 0.01:
        severity = "HIGH"
    elif ks_pvalue < 0.05:
        severity = "MEDIUM"
    else:
        severity = "LOW"
    
    drift_results.append({
        'Feature': feature,
        'KS_Statistic': ks_statistic,
        'KS_P_Value': ks_pvalue,
        'Drift_Detected': drift_detected,
        'Severity': severity,
        'Reference_Mean': ref_mean,
        'Current_Mean': curr_mean,
        'Mean_Change_%': mean_change,
        'Reference_Std': ref_std,
        'Current_Std': curr_std,
        'Std_Change_%': std_change
    })

# Create drift report DataFrame
drift_df = pd.DataFrame(drift_results)

# Sort by severity and p-value
drift_df = drift_df.sort_values(['Drift_Detected', 'KS_P_Value'], ascending=[False, True])

# Save drift report
drift_report_path = 'reports/drift/drift_report.csv'
drift_df.to_csv(drift_report_path, index=False)

print(f"✓ Analyzed {len(numerical_features)} features")
print(f"✓ Drift detected in {drift_df['Drift_Detected'].sum()} features")

# Display drift summary
print("\n" + "=" * 80)
print("DRIFT DETECTION SUMMARY")
print("=" * 80)

drift_detected_features = drift_df[drift_df['Drift_Detected'] == True]

if len(drift_detected_features) > 0:
    print(f"\n⚠️  DRIFT DETECTED in {len(drift_detected_features)} features:")
    print()
    
    for _, row in drift_detected_features.iterrows():
        print(f"📊 {row['Feature']}")
        print(f"   Severity: {row['Severity']}")
        print(f"   KS p-value: {row['KS_P_Value']:.6f}")
        print(f"   Mean change: {row['Mean_Change_%']:+.2f}%")
        print(f"   Std change: {row['Std_Change_%']:+.2f}%")
        print()
else:
    print("✅ No significant drift detected in any features")

# ============================================================================
# 4. VISUALIZE DRIFT
# ============================================================================

print("\n[STEP 4] Creating drift visualizations...")
print("-" * 80)

# 4.1: Distribution comparison plots for drifted features
drifted_features = drift_df[drift_df['Drift_Detected'] == True]['Feature'].tolist()

if len(drifted_features) > 0:
    # Plot top 6 drifted features
    plot_features = drifted_features[:6]
    
    n_plots = len(plot_features)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, feature in enumerate(plot_features):
        ax = axes[idx]
        
        # Get data
        ref_values = reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        # Plot distributions
        ax.hist(ref_values, bins=30, alpha=0.6, label='Reference', color='blue', density=True)
        ax.hist(curr_values, bins=30, alpha=0.6, label='Current', color='red', density=True)
        
        # Get drift info
        drift_info = drift_df[drift_df['Feature'] == feature].iloc[0]
        
        ax.set_title(f"{feature}\n(p={drift_info['KS_P_Value']:.4f}, Severity: {drift_info['Severity']})",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/drift/plots/distribution_drift.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/drift/plots/distribution_drift.png")
    plt.close()

# 4.2: Drift severity heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Create severity matrix for heatmap
severity_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
drift_df['Severity_Code'] = drift_df['Severity'].map(severity_map)

# Sort by severity
drift_sorted = drift_df.sort_values('Severity_Code', ascending=False)

# Create color-coded bars
colors = ['green' if not d else 'orange' if s == 'MEDIUM' else 'red' 
          for d, s in zip(drift_sorted['Drift_Detected'], drift_sorted['Severity'])]

ax.barh(drift_sorted['Feature'], drift_sorted['KS_Statistic'], color=colors, alpha=0.7)
ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Medium Threshold')
ax.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='High Threshold')

ax.set_xlabel('KS Statistic (Higher = More Drift)', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Data Drift Detection - All Features', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/drift/plots/drift_severity.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/drift/plots/drift_severity.png")
plt.close()

# 4.3: Mean and Std change plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean change
ax1 = axes[0]
colors1 = ['red' if abs(x) > 10 else 'orange' if abs(x) > 5 else 'green' 
           for x in drift_df['Mean_Change_%']]

ax1.barh(drift_df['Feature'], drift_df['Mean_Change_%'], color=colors1, alpha=0.7)
ax1.axvline(x=0, color='black', linewidth=1)
ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.5)
ax1.axvline(x=-10, color='orange', linestyle='--', alpha=0.5)
ax1.set_xlabel('Mean Change (%)', fontsize=11, fontweight='bold')
ax1.set_title('Mean Value Drift', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Std change
ax2 = axes[1]
colors2 = ['red' if abs(x) > 10 else 'orange' if abs(x) > 5 else 'green' 
           for x in drift_df['Std_Change_%']]

ax2.barh(drift_df['Feature'], drift_df['Std_Change_%'], color=colors2, alpha=0.7)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.axvline(x=10, color='orange', linestyle='--', alpha=0.5)
ax2.axvline(x=-10, color='orange', linestyle='--', alpha=0.5)
ax2.set_xlabel('Std Deviation Change (%)', fontsize=11, fontweight='bold')
ax2.set_title('Variance Drift', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/drift/plots/statistical_changes.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/drift/plots/statistical_changes.png")
plt.close()

# ============================================================================
# 5. TEMPORAL DRIFT ANALYSIS
# ============================================================================

print("\n[STEP 5] Analyzing temporal drift patterns...")
print("-" * 80)

# Use rolling windows to detect when drift started
window_size = 30  # 30-day windows

if len(drifted_features) > 0:
    # Analyze first drifted feature
    feature = drifted_features[0]
    
    # Calculate rolling statistics
    merged_data['rolling_mean'] = merged_data[feature].rolling(window=window_size).mean()
    merged_data['rolling_std'] = merged_data[feature].rolling(window=window_size).std()
    
    # Calculate reference statistics
    ref_mean = reference_data[feature].mean()
    ref_std = reference_data[feature].std()
    
    # Plot temporal drift
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Raw values with rolling mean
    ax1 = axes[0]
    ax1.scatter(merged_data['date'], merged_data[feature], alpha=0.3, s=10, label='Daily Values')
    ax1.plot(merged_data['date'], merged_data['rolling_mean'], 
            color='red', linewidth=2, label=f'{window_size}-day Moving Average')
    ax1.axhline(y=ref_mean, color='blue', linestyle='--', linewidth=2, label='Reference Mean')
    ax1.axvline(x=reference_data['date'].max(), color='green', linestyle='--', 
               linewidth=2, label='Train/Monitor Split')
    ax1.set_ylabel(feature, fontsize=11, fontweight='bold')
    ax1.set_title(f'Temporal Analysis - {feature}', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Rolling std
    ax2 = axes[1]
    ax2.plot(merged_data['date'], merged_data['rolling_std'], 
            color='purple', linewidth=2, label=f'{window_size}-day Rolling Std')
    ax2.axhline(y=ref_std, color='blue', linestyle='--', linewidth=2, label='Reference Std')
    ax2.axvline(x=reference_data['date'].max(), color='green', linestyle='--', 
               linewidth=2, label='Train/Monitor Split')
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    ax2.set_title('Variance Over Time', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/drift/plots/temporal_drift.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/drift/plots/temporal_drift.png")
    plt.close()

# ============================================================================
# 6. GENERATE DRIFT REPORT
# ============================================================================

print("\n[STEP 6] Generating drift detection report...")
print("-" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("DATA DRIFT DETECTION REPORT")
report_lines.append("Health Risk Prediction System")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("\n" + "=" * 80)

# Executive Summary
report_lines.append("\n1. EXECUTIVE SUMMARY")
report_lines.append("-" * 80)

total_features = len(drift_df)
drifted_count = drift_df['Drift_Detected'].sum()
drift_percentage = (drifted_count / total_features) * 100

report_lines.append(f"\nTotal Features Monitored: {total_features}")
report_lines.append(f"Features with Drift Detected: {drifted_count} ({drift_percentage:.1f}%)")
report_lines.append(f"Features Stable: {total_features - drifted_count}")

high_severity = len(drift_df[drift_df['Severity'] == 'HIGH'])
medium_severity = len(drift_df[drift_df['Severity'] == 'MEDIUM'])

report_lines.append(f"\nDrift Severity Breakdown:")
report_lines.append(f"  • HIGH severity: {high_severity}")
report_lines.append(f"  • MEDIUM severity: {medium_severity}")
report_lines.append(f"  • LOW/No drift: {total_features - high_severity - medium_severity}")

# Data Windows
report_lines.append("\n\n2. DATA WINDOWS")
report_lines.append("-" * 80)
report_lines.append(f"\nReference Period (Training):")
report_lines.append(f"  Date Range: {reference_data['date'].min()} to {reference_data['date'].max()}")
report_lines.append(f"  Records: {len(reference_data)}")

report_lines.append(f"\nCurrent Period (Monitoring):")
report_lines.append(f"  Date Range: {current_data['date'].min()} to {current_data['date'].max()}")
report_lines.append(f"  Records: {len(current_data)}")

# Drift Details
report_lines.append("\n\n3. DETAILED DRIFT ANALYSIS")
report_lines.append("-" * 80)

if drifted_count > 0:
    report_lines.append("\n⚠️ FEATURES WITH DETECTED DRIFT:\n")
    
    for _, row in drift_detected_features.iterrows():
        report_lines.append(f"\n{row['Feature']}")
        report_lines.append(f"  Severity: {row['Severity']}")
        report_lines.append(f"  KS Statistic: {row['KS_Statistic']:.4f}")
        report_lines.append(f"  P-Value: {row['KS_P_Value']:.6f}")
        report_lines.append(f"  Reference Mean: {row['Reference_Mean']:.4f}")
        report_lines.append(f"  Current Mean: {row['Current_Mean']:.4f}")
        report_lines.append(f"  Mean Change: {row['Mean_Change_%']:+.2f}%")
        report_lines.append(f"  Std Change: {row['Std_Change_%']:+.2f}%")
else:
    report_lines.append("\n✅ No significant drift detected in any features")

# Stable Features
stable_features = drift_df[drift_df['Drift_Detected'] == False]
if len(stable_features) > 0:
    report_lines.append("\n\n✅ STABLE FEATURES (No Drift):\n")
    for feature in stable_features['Feature'].head(10):
        report_lines.append(f"  • {feature}")

# Recommendations
report_lines.append("\n\n4. RECOMMENDATIONS")
report_lines.append("-" * 80)

if drifted_count > 0:
    report_lines.append("\n⚠️ ACTIONS REQUIRED:")
    report_lines.append("\n1. Model Retraining:")
    report_lines.append("   • Significant drift detected - model retraining recommended")
    report_lines.append("   • Use most recent data for retraining")
    report_lines.append("   • Consider expanding training dataset")
    
    report_lines.append("\n2. Data Quality Check:")
    report_lines.append("   • Verify data collection processes")
    report_lines.append("   • Check for sensor calibration issues")
    report_lines.append("   • Investigate sudden distribution changes")
    
    report_lines.append("\n3. Monitoring:")
    report_lines.append("   • Increase monitoring frequency")
    report_lines.append("   • Set up automated drift alerts")
    report_lines.append("   • Track model performance metrics")
else:
    report_lines.append("\n✅ NO IMMEDIATE ACTION REQUIRED:")
    report_lines.append("   • Data distributions remain stable")
    report_lines.append("   • Continue regular monitoring")
    report_lines.append("   • Schedule next drift check in 30 days")

# Methodology
report_lines.append("\n\n5. METHODOLOGY")
report_lines.append("-" * 80)
report_lines.append("\nDrift Detection Method: Kolmogorov-Smirnov Test")
report_lines.append("  • Tests if two samples come from same distribution")
report_lines.append("  • Significance level: p < 0.05")
report_lines.append("  • Non-parametric test (no distribution assumptions)")

report_lines.append("\nSeverity Classification:")
report_lines.append("  • HIGH: p-value < 0.01 (very significant drift)")
report_lines.append("  • MEDIUM: p-value < 0.05 (significant drift)")
report_lines.append("  • LOW: p-value >= 0.05 (no significant drift)")

report_lines.append("\n" + "=" * 80)
report_lines.append("END OF DRIFT REPORT")
report_lines.append("=" * 80)

# Save report
report_text = '\n'.join(report_lines)
report_file = 'reports/drift/drift_detection_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"✓ Report saved: {report_file}")

# Print summary to console
print("\n" + report_text)

# ============================================================================
# 7. SAVE DRIFT METADATA
# ============================================================================

print("\n[STEP 7] Saving drift detection metadata...")
print("-" * 80)

metadata = {
    'detection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'reference_period': {
        'start': str(reference_data['date'].min()),
        'end': str(reference_data['date'].max()),
        'records': len(reference_data)
    },
    'current_period': {
        'start': str(current_data['date'].min()),
        'end': str(current_data['date'].max()),
        'records': len(current_data)
    },
    'summary': {
        'total_features': total_features,
        'drifted_features': int(drifted_count),
        'drift_percentage': float(drift_percentage),
        'high_severity_count': int(high_severity),
        'medium_severity_count': int(medium_severity)
    },
    'drifted_features_list': drifted_features
}

metadata_path = 'reports/drift/drift_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("✅ DATA DRIFT DETECTION COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  Reports:")
print("    - reports/drift/drift_report.csv")
print("    - reports/drift/drift_detection_report.txt")
print("    - reports/drift/drift_metadata.pkl")
print("\n  Visualizations:")
print("    - reports/drift/plots/distribution_drift.png")
print("    - reports/drift/plots/drift_severity.png")
print("    - reports/drift/plots/statistical_changes.png")
print("    - reports/drift/plots/temporal_drift.png")

print("\nKey Findings:")
print(f"  • Drift detected in {drifted_count}/{total_features} features ({drift_percentage:.1f}%)")
if drifted_count > 0:
    print(f"  • HIGH severity drift: {high_severity} features")
    print(f"  • MEDIUM severity drift: {medium_severity} features")
    print(f"  ⚠️  Model retraining recommended")
else:
    print(f"  ✅ All features stable - no retraining needed")

print("\nNext Steps:")
print("  1. ✅ Data drift detection complete")
print("  2. ⏭️  Review drift report and visualizations")
print("  3. ⏭️  If drift detected, retrain models with recent data")
print("  4. ⏭️  Set up automated drift monitoring")

print("=" * 80)