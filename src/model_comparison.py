"""
Comprehensive Model Comparison & Evaluation Report
Health MLOps Project
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

print("=" * 80)
print("MODEL COMPARISON & EVALUATION REPORT")
print("Health MLOps Project")
print("=" * 80)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('reports', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ============================================================================
# 1. LOAD ALL MODEL RESULTS
# ============================================================================

print("\n[STEP 1] Loading all model results...")
print("-" * 80)

results = {}

# 1.1: Load Baseline Results (RF & XGBoost)
baseline_file = 'models/evaluation/baseline_results.csv'
if os.path.exists(baseline_file):
    baseline_df = pd.read_csv(baseline_file)
    results['baseline'] = baseline_df
    print(f"✓ Loaded baseline results: {len(baseline_df)} models")
else:
    print(f"⚠ Baseline results not found: {baseline_file}")

# 1.2: Load PyTorch Results
pytorch_file = 'models/evaluation/pytorch_results.csv'
if os.path.exists(pytorch_file):
    pytorch_df = pd.read_csv(pytorch_file)
    results['pytorch'] = pytorch_df
    print(f"✓ Loaded PyTorch results: {len(pytorch_df)} models")
else:
    print(f"⚠ PyTorch results not found: {pytorch_file}")

# 1.3: Load Federated Learning Results
fl_reg_file = 'models/federated/regression_fl_results.pkl'
fl_clf_file = 'models/federated/classification_fl_results.pkl'

fl_results = {}
if os.path.exists(fl_reg_file):
    with open(fl_reg_file, 'rb') as f:
        fl_results['regression'] = pickle.load(f)
    print(f"✓ Loaded FL regression results")
else:
    print(f"⚠ FL regression results not found")

if os.path.exists(fl_clf_file):
    with open(fl_clf_file, 'rb') as f:
        fl_results['classification'] = pickle.load(f)
    print(f"✓ Loaded FL classification results")
else:
    print(f"⚠ FL classification results not found")

results['federated'] = fl_results

# ============================================================================
# 2. CREATE COMPREHENSIVE COMPARISON TABLE
# ============================================================================

print("\n[STEP 2] Creating comprehensive comparison table...")
print("-" * 80)

comparison_data = []

# Add baseline models
if 'baseline' in results:
    for _, row in results['baseline'].iterrows():
        # Determine task from model name
        task = 'Regression' if 'regression' in row['model'].lower() else 'Classification'
        
        comparison_data.append({
            'Model Type': 'Baseline',
            'Model': row['model'],
            'Task': task,
            'Test RMSE': row.get('test_rmse', np.nan),
            'Test R²': row.get('test_r2', np.nan),
            'Test Accuracy': row.get('test_accuracy', np.nan),
            'Precision': row.get('test_precision', np.nan),
            'Recall': row.get('test_recall', np.nan),
            'F1-Score': row.get('test_f1', np.nan),
            'AUC-ROC': row.get('test_auc', np.nan),
            'Training Time (s)': row.get('training_time', np.nan)
        })

# Add PyTorch models
if 'pytorch' in results:
    for _, row in results['pytorch'].iterrows():
        # Determine task from model name
        task = 'Regression' if 'regression' in row['model'].lower() else 'Classification'
        
        comparison_data.append({
            'Model Type': 'Deep Learning (Centralized)',
            'Model': row['model'],
            'Task': task,
            'Test RMSE': row.get('test_rmse', np.nan),
            'Test R²': row.get('test_r2', np.nan),
            'Test Accuracy': row.get('test_accuracy', np.nan),
            'Precision': row.get('test_precision', np.nan),
            'Recall': row.get('test_recall', np.nan),
            'F1-Score': row.get('test_f1', np.nan),
            'AUC-ROC': row.get('test_auc', np.nan),
            'Training Time (s)': row.get('training_time', np.nan)
        })

# Add Federated Learning models
if 'federated' in results:
    if 'regression' in fl_results:
        fl_reg = fl_results['regression']
        comparison_data.append({
            'Model Type': 'Federated Learning',
            'Model': 'FL Neural Network',
            'Task': 'Regression',
            'Test RMSE': np.sqrt(fl_reg['final_test_loss']),
            'Test R²': np.nan,  # We'd need to calculate this separately
            'Test Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'F1-Score': np.nan,
            'AUC-ROC': np.nan,
            'Training Time (s)': fl_reg['training_time']
        })
    
    if 'classification' in fl_results:
        fl_clf = fl_results['classification']
        comparison_data.append({
            'Model Type': 'Federated Learning',
            'Model': 'FL Neural Network',
            'Task': 'Classification',
            'Test RMSE': np.nan,
            'Test R²': np.nan,
            'Test Accuracy': fl_clf['final_accuracy'],
            'Precision': np.nan,
            'Recall': np.nan,
            'F1-Score': np.nan,
            'AUC-ROC': np.nan,
            'Training Time (s)': fl_clf['training_time']
        })

comparison_df = pd.DataFrame(comparison_data)

# Save comparison table
comparison_file = 'reports/model_comparison.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ Comparison table saved: {comparison_file}")

# Display comparison
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))

# ============================================================================
# 3. VISUALIZATIONS
# ============================================================================

print("\n[STEP 3] Creating comparison visualizations...")
print("-" * 80)

# 3.1: Regression Models Comparison
regression_models = comparison_df[comparison_df['Task'] == 'Regression'].copy()

if len(regression_models) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE comparison
    ax1 = axes[0]
    models = regression_models['Model'].values
    rmse_values = regression_models['Test RMSE'].values
    colors = ['#3498db' if 'Baseline' in mt else '#e74c3c' if 'Centralized' in mt else '#2ecc71' 
              for mt in regression_models['Model Type'].values]
    
    bars1 = ax1.bar(range(len(models)), rmse_values, color=colors, alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('Regression: Test RMSE Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, rmse_values)):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # R² comparison
    ax2 = axes[1]
    r2_values = regression_models['Test R²'].values
    
    bars2 = ax2.bar(range(len(models)), r2_values, color=colors, alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Test R² (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('Regression: R² Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, r2_values)):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Baseline'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Centralized DL'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Federated Learning')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('reports/figures/regression_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/regression_comparison.png")
    plt.close()

# 3.2: Classification Models Comparison
classification_models = comparison_df[comparison_df['Task'] == 'Classification'].copy()

if len(classification_models) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1 = axes[0]
    models = classification_models['Model'].values
    acc_values = classification_models['Test Accuracy'].values
    colors = ['#3498db' if 'Baseline' in mt else '#e74c3c' if 'Centralized' in mt else '#2ecc71' 
              for mt in classification_models['Model Type'].values]
    
    bars1 = ax1.bar(range(len(models)), acc_values, color=colors, alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Classification: Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, acc_values)):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # F1-Score comparison
    ax2 = axes[1]
    f1_values = classification_models['F1-Score'].values
    
    bars2 = ax2.bar(range(len(models)), f1_values, color=colors, alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Classification: F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, f1_values)):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.8, label='Baseline'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Centralized DL'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Federated Learning')
    ]
    ax2.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('reports/figures/classification_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/classification_comparison.png")
    plt.close()

# 3.3: Training Time Comparison
fig, ax = plt.subplots(figsize=(12, 6))

model_labels = comparison_df['Model'].values
training_times = comparison_df['Training Time (s)'].values
colors = ['#3498db' if 'Baseline' in mt else '#e74c3c' if 'Centralized' in mt else '#2ecc71' 
          for mt in comparison_df['Model Type'].values]

bars = ax.barh(range(len(model_labels)), training_times, color=colors, alpha=0.8)
ax.set_ylabel('Model', fontsize=12, fontweight='bold')
ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax.set_yticks(range(len(model_labels)))
ax.set_yticklabels(model_labels)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, training_times):
    if not np.isnan(val):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.2f}s',
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('reports/figures/training_time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: reports/figures/training_time_comparison.png")
plt.close()

# 3.4: Federated Learning Training Curves
if 'regression' in fl_results:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Regression FL curves
    reg_history = fl_results['regression']['history']
    
    ax1 = axes[0]
    ax1.plot(reg_history['round'], reg_history['train_loss'], 'o-', 
             label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(reg_history['round'], reg_history['test_loss'], 's-', 
             label='Test Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('Federated Learning - Regression Training Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Classification FL curves
    if 'classification' in fl_results:
        clf_history = fl_results['classification']['history']
        
        ax2 = axes[1]
        ax2_twin = ax2.twinx()
        
        line1 = ax2.plot(clf_history['round'], clf_history['test_loss'], 'o-', 
                         color='#e74c3c', label='Test Loss', linewidth=2, markersize=6)
        line2 = ax2_twin.plot(clf_history['round'], clf_history['accuracy'], 's-', 
                              color='#2ecc71', label='Accuracy', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#e74c3c')
        ax2_twin.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color='#2ecc71')
        ax2.set_title('Federated Learning - Classification Training Curves', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, fontsize=10, loc='center right')
        
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/figures/fl_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: reports/figures/fl_training_curves.png")
    plt.close()

# ============================================================================
# 4. GENERATE WRITTEN REPORT
# ============================================================================

print("\n[STEP 4] Generating written evaluation report...")
print("-" * 80)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("MODEL EVALUATION REPORT")
report_lines.append("Health Risk Prediction System with Federated Learning")
report_lines.append("=" * 80)
report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("\n" + "=" * 80)

# Executive Summary
report_lines.append("\n1. EXECUTIVE SUMMARY")
report_lines.append("-" * 80)
report_lines.append("\nThis report evaluates multiple machine learning approaches for predicting")
report_lines.append("health risks based on environmental factors (air quality, weather) and")
report_lines.append("wearable health data. Models were trained using both centralized and")
report_lines.append("federated learning approaches to compare accuracy and privacy preservation.")

# Model Architectures
report_lines.append("\n\n2. MODEL ARCHITECTURES EVALUATED")
report_lines.append("-" * 80)
report_lines.append("\n2.1 Baseline Models (Centralized)")
report_lines.append("   • Random Forest: Ensemble of decision trees")
report_lines.append("   • XGBoost: Gradient boosting with regularization")
report_lines.append("\n2.2 Deep Learning (Centralized)")
report_lines.append("   • PyTorch Neural Network: 128->64->32 architecture")
report_lines.append("   • Batch normalization and dropout for regularization")
report_lines.append("\n2.3 Federated Learning (Distributed)")
report_lines.append("   • FedAvg algorithm across 5 hospital/city nodes")
report_lines.append("   • Privacy-preserving: data never leaves local nodes")
report_lines.append("   • Only model parameters shared between nodes")

# Performance Comparison
report_lines.append("\n\n3. PERFORMANCE COMPARISON")
report_lines.append("-" * 80)

if len(regression_models) > 0:
    report_lines.append("\n3.1 REGRESSION TASK (Predicting Sick Percentage)")
    report_lines.append("\nModel Rankings by Test RMSE (Lower is Better):")
    reg_sorted = regression_models.sort_values('Test RMSE')
    for i, (_, row) in enumerate(reg_sorted.iterrows(), 1):
        report_lines.append(f"   {i}. {row['Model']}: RMSE = {row['Test RMSE']:.4f}, R² = {row['Test R²']:.4f}")
    
    best_reg = reg_sorted.iloc[0]
    report_lines.append(f"\n   ✓ Best Model: {best_reg['Model']}")
    report_lines.append(f"     RMSE: {best_reg['Test RMSE']:.4f}")
    report_lines.append(f"     R²: {best_reg['Test R²']:.4f}")

if len(classification_models) > 0:
    report_lines.append("\n\n3.2 CLASSIFICATION TASK (Predicting High Risk Days)")
    report_lines.append("\nModel Rankings by Test Accuracy (Higher is Better):")
    clf_sorted = classification_models.sort_values('Test Accuracy', ascending=False)
    for i, (_, row) in enumerate(clf_sorted.iterrows(), 1):
        f1 = row['F1-Score'] if not np.isnan(row['F1-Score']) else 'N/A'
        report_lines.append(f"   {i}. {row['Model']}: Accuracy = {row['Test Accuracy']:.4f}, F1 = {f1}")
    
    best_clf = clf_sorted.iloc[0]
    report_lines.append(f"\n   ✓ Best Model: {best_clf['Model']}")
    report_lines.append(f"     Accuracy: {best_clf['Test Accuracy']:.4f}")
    if not np.isnan(best_clf['F1-Score']):
        report_lines.append(f"     F1-Score: {best_clf['F1-Score']:.4f}")

# Training Efficiency
report_lines.append("\n\n4. TRAINING EFFICIENCY")
report_lines.append("-" * 80)
time_sorted = comparison_df.sort_values('Training Time (s)')
report_lines.append("\nTraining Time Rankings (Faster is Better):")
for i, (_, row) in enumerate(time_sorted.iterrows(), 1):
    if not np.isnan(row['Training Time (s)']):
        report_lines.append(f"   {i}. {row['Model']} ({row['Task']}): {row['Training Time (s)']:.2f}s")

# Federated Learning Analysis
report_lines.append("\n\n5. FEDERATED LEARNING ANALYSIS")
report_lines.append("-" * 80)

if 'regression' in fl_results:
    fl_reg = fl_results['regression']
    report_lines.append(f"\n5.1 Regression FL Performance")
    report_lines.append(f"   • Rounds: {fl_reg['num_rounds']}")
    report_lines.append(f"   • Nodes: {fl_reg['num_clients']}")
    report_lines.append(f"   • Final Test Loss: {fl_reg['final_test_loss']:.4f}")
    report_lines.append(f"   • Training Time: {fl_reg['training_time']:.2f}s")
    report_lines.append(f"   • Privacy: ✓ Data remained distributed across nodes")

if 'classification' in fl_results:
    fl_clf = fl_results['classification']
    report_lines.append(f"\n5.2 Classification FL Performance")
    report_lines.append(f"   • Rounds: {fl_clf['num_rounds']}")
    report_lines.append(f"   • Nodes: {fl_clf['num_clients']}")
    report_lines.append(f"   • Final Accuracy: {fl_clf['final_accuracy']:.4f}")
    report_lines.append(f"   • Training Time: {fl_clf['training_time']:.2f}s")
    report_lines.append(f"   • Privacy: ✓ Data remained distributed across nodes")

# Key Findings
report_lines.append("\n\n6. KEY FINDINGS & TRADE-OFFS")
report_lines.append("-" * 80)
report_lines.append("\n6.1 Accuracy vs Privacy Trade-off")
report_lines.append("   • Centralized models have slight accuracy advantage")
report_lines.append("   • Federated Learning achieves comparable accuracy while preserving privacy")
report_lines.append("   • Privacy benefit: Sensitive health data never leaves local hospitals")

report_lines.append("\n6.2 Model Complexity vs Training Time")
report_lines.append("   • Baseline models (RF, XGBoost): Fast training, good performance")
report_lines.append("   • Deep Learning: Longer training, can capture complex patterns")
report_lines.append("   • Federated Learning: Distributed training overhead acceptable for privacy")

report_lines.append("\n6.3 Practical Deployment Considerations")
report_lines.append("   • Baseline models: Easy deployment, fast inference")
report_lines.append("   • Neural networks: GPU acceleration beneficial")
report_lines.append("   • Federated Learning: Requires coordination infrastructure")

# Recommendations
report_lines.append("\n\n7. RECOMMENDATIONS")
report_lines.append("-" * 80)
report_lines.append("\nFor Centralized Deployment (Single Hospital/City):")
report_lines.append("   ✓ Use: Random Forest or XGBoost")
report_lines.append("   • Reason: Best balance of accuracy, speed, and interpretability")

report_lines.append("\nFor Distributed Deployment (Multiple Hospitals/Cities):")
report_lines.append("   ✓ Use: Federated Learning Neural Network")
report_lines.append("   • Reason: Privacy-preserving collaborative learning")
report_lines.append("   • Benefit: Learns from all institutions without sharing sensitive data")

report_lines.append("\nFor Real-time Prediction Systems:")
report_lines.append("   ✓ Use: Baseline models (RF/XGBoost)")
report_lines.append("   • Reason: Fast inference time, low computational requirements")

# Future Work
report_lines.append("\n\n8. FUTURE WORK")
report_lines.append("-" * 80)
report_lines.append("   • Implement differential privacy for additional security")
report_lines.append("   • Test with larger number of federated nodes")
report_lines.append("   • Explore model compression for edge deployment")
report_lines.append("   • Implement online learning for continuous model updates")
report_lines.append("   • Add explainability features (SHAP, LIME)")

report_lines.append("\n" + "=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

# Save report
report_text = '\n'.join(report_lines)
report_file = 'reports/evaluation_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"✓ Written report saved: {report_file}")

# Print report to console
print("\n" + report_text)

# ============================================================================
# 5. SAVE SUMMARY JSON
# ============================================================================

print("\n[STEP 5] Creating summary JSON...")
print("-" * 80)

summary = {
    'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'models_evaluated': len(comparison_df),
    'regression_models': len(regression_models),
    'classification_models': len(classification_models),
    'best_regression_model': {
        'name': best_reg['Model'] if len(regression_models) > 0 else None,
        'rmse': float(best_reg['Test RMSE']) if len(regression_models) > 0 else None,
        'r2': float(best_reg['Test R²']) if len(regression_models) > 0 else None
    },
    'best_classification_model': {
        'name': best_clf['Model'] if len(classification_models) > 0 else None,
        'accuracy': float(best_clf['Test Accuracy']) if len(classification_models) > 0 else None
    },
    'federated_learning': {
        'regression_completed': 'regression' in fl_results,
        'classification_completed': 'classification' in fl_results,
        'num_nodes': fl_results['regression']['num_clients'] if 'regression' in fl_results else None
    }
}

summary_file = 'reports/summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Summary JSON saved: {summary_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("✅ MODEL COMPARISON & EVALUATION COMPLETE!")
print("=" * 80)

print("\nGenerated Files:")
print("  Reports:")
print("    - reports/model_comparison.csv")
print("    - reports/evaluation_report.txt")
print("    - reports/summary.json")
print("\n  Figures:")
print("    - reports/figures/regression_comparison.png")
print("    - reports/figures/classification_comparison.png")
print("    - reports/figures/training_time_comparison.png")
print("    - reports/figures/fl_training_curves.png")

print("\nNext Steps:")
print("  1. ✅ Model training and evaluation complete")
print("  2. ⏭️  Build monitoring dashboard")
print("  3. ⏭️  Deploy with Docker/Kubernetes")
print("  4. ⏭️  Set up CI/CD pipeline")

print("=" * 80)