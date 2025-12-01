"""
Baseline Models Training - Health MLOps Project
Random Forest and XGBoost for Regression and Classification
"""

import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("BASELINE MODELS TRAINING - Health MLOps Project")
print("=" * 80)

# Create directories
os.makedirs('models/baseline', exist_ok=True)
os.makedirs('models/evaluation', exist_ok=True)
os.makedirs('models/plots', exist_ok=True)

# ============================================================================
# STEP 1: Load Centralized Data
# ============================================================================
print("\n[STEP 1] Loading centralized dataset...")
print("-" * 80)

with open('data/processed/centralized_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_reg_train = data['y_regression_train']
y_reg_test = data['y_regression_test']
y_cls_train = data['y_classification_train']
y_cls_test = data['y_classification_test']
feature_names = data['feature_names']

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Number of features: {len(feature_names)}")
print(f"✓ Regression target: sick_percentage")
print(f"✓ Classification target: high_risk (binary)")

# Check class distribution
print(f"\nClass distribution (train):")
print(f"  Low risk (0): {(y_cls_train == 0).sum()} ({(y_cls_train == 0).mean()*100:.1f}%)")
print(f"  High risk (1): {(y_cls_train == 1).sum()} ({(y_cls_train == 1).mean()*100:.1f}%)")

# ============================================================================
# STEP 2: Train Random Forest Models
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 2] Training Random Forest Models")
print("=" * 80)

results = {}

# -------- Random Forest Regression --------
print("\n[2.1] Random Forest - Regression Task")
print("-" * 80)

print("Training Random Forest Regressor...")
start_time = time.time()

rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_reg.fit(X_train, y_reg_train)
train_time_rf_reg = time.time() - start_time

print(f"✓ Training completed in {train_time_rf_reg:.2f} seconds")

# Predictions
y_reg_pred_train = rf_reg.predict(X_train)
y_reg_pred_test = rf_reg.predict(X_test)

# Metrics
rf_reg_metrics = {
    'model': 'Random Forest Regression',
    'train_rmse': np.sqrt(mean_squared_error(y_reg_train, y_reg_pred_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test)),
    'train_mae': mean_absolute_error(y_reg_train, y_reg_pred_train),
    'test_mae': mean_absolute_error(y_reg_test, y_reg_pred_test),
    'train_r2': r2_score(y_reg_train, y_reg_pred_train),
    'test_r2': r2_score(y_reg_test, y_reg_pred_test),
    'training_time': train_time_rf_reg
}

print(f"\nRegression Metrics:")
print(f"  Train RMSE: {rf_reg_metrics['train_rmse']:.4f}")
print(f"  Test RMSE:  {rf_reg_metrics['test_rmse']:.4f}")
print(f"  Train MAE:  {rf_reg_metrics['train_mae']:.4f}")
print(f"  Test MAE:   {rf_reg_metrics['test_mae']:.4f}")
print(f"  Train R²:   {rf_reg_metrics['train_r2']:.4f}")
print(f"  Test R²:    {rf_reg_metrics['test_r2']:.4f}")

results['rf_regression'] = rf_reg_metrics

# Save model
with open('models/baseline/rf_regressor.pkl', 'wb') as f:
    pickle.dump(rf_reg, f)
print("✓ Saved: models/baseline/rf_regressor.pkl")

# -------- Random Forest Classification --------
print("\n[2.2] Random Forest - Classification Task")
print("-" * 80)

print("Training Random Forest Classifier...")
start_time = time.time()

rf_cls = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_cls.fit(X_train, y_cls_train)
train_time_rf_cls = time.time() - start_time

print(f"✓ Training completed in {train_time_rf_cls:.2f} seconds")

# Predictions
y_cls_pred_train = rf_cls.predict(X_train)
y_cls_pred_test = rf_cls.predict(X_test)
y_cls_proba_test = rf_cls.predict_proba(X_test)[:, 1]

# Metrics
rf_cls_metrics = {
    'model': 'Random Forest Classification',
    'train_accuracy': accuracy_score(y_cls_train, y_cls_pred_train),
    'test_accuracy': accuracy_score(y_cls_test, y_cls_pred_test),
    'train_precision': precision_score(y_cls_train, y_cls_pred_train, zero_division=0),
    'test_precision': precision_score(y_cls_test, y_cls_pred_test, zero_division=0),
    'train_recall': recall_score(y_cls_train, y_cls_pred_train, zero_division=0),
    'test_recall': recall_score(y_cls_test, y_cls_pred_test, zero_division=0),
    'train_f1': f1_score(y_cls_train, y_cls_pred_train, zero_division=0),
    'test_f1': f1_score(y_cls_test, y_cls_pred_test, zero_division=0),
    'test_auc': roc_auc_score(y_cls_test, y_cls_proba_test),
    'training_time': train_time_rf_cls
}

print(f"\nClassification Metrics:")
print(f"  Train Accuracy:  {rf_cls_metrics['train_accuracy']:.4f}")
print(f"  Test Accuracy:   {rf_cls_metrics['test_accuracy']:.4f}")
print(f"  Test Precision:  {rf_cls_metrics['test_precision']:.4f}")
print(f"  Test Recall:     {rf_cls_metrics['test_recall']:.4f}")
print(f"  Test F1-Score:   {rf_cls_metrics['test_f1']:.4f}")
print(f"  Test AUC-ROC:    {rf_cls_metrics['test_auc']:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_cls_test, y_cls_pred_test)
print(cm)

results['rf_classification'] = rf_cls_metrics

# Save model
with open('models/baseline/rf_classifier.pkl', 'wb') as f:
    pickle.dump(rf_cls, f)
print("✓ Saved: models/baseline/rf_classifier.pkl")

# ============================================================================
# STEP 3: Train XGBoost Models
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 3] Training XGBoost Models")
print("=" * 80)

# -------- XGBoost Regression --------
print("\n[3.1] XGBoost - Regression Task")
print("-" * 80)

print("Training XGBoost Regressor...")
start_time = time.time()

xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_reg.fit(X_train, y_reg_train)
train_time_xgb_reg = time.time() - start_time

print(f"✓ Training completed in {train_time_xgb_reg:.2f} seconds")

# Predictions
y_reg_pred_train_xgb = xgb_reg.predict(X_train)
y_reg_pred_test_xgb = xgb_reg.predict(X_test)

# Metrics
xgb_reg_metrics = {
    'model': 'XGBoost Regression',
    'train_rmse': np.sqrt(mean_squared_error(y_reg_train, y_reg_pred_train_xgb)),
    'test_rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test_xgb)),
    'train_mae': mean_absolute_error(y_reg_train, y_reg_pred_train_xgb),
    'test_mae': mean_absolute_error(y_reg_test, y_reg_pred_test_xgb),
    'train_r2': r2_score(y_reg_train, y_reg_pred_train_xgb),
    'test_r2': r2_score(y_reg_test, y_reg_pred_test_xgb),
    'training_time': train_time_xgb_reg
}

print(f"\nRegression Metrics:")
print(f"  Train RMSE: {xgb_reg_metrics['train_rmse']:.4f}")
print(f"  Test RMSE:  {xgb_reg_metrics['test_rmse']:.4f}")
print(f"  Train MAE:  {xgb_reg_metrics['train_mae']:.4f}")
print(f"  Test MAE:   {xgb_reg_metrics['test_mae']:.4f}")
print(f"  Train R²:   {xgb_reg_metrics['train_r2']:.4f}")
print(f"  Test R²:    {xgb_reg_metrics['test_r2']:.4f}")

results['xgb_regression'] = xgb_reg_metrics

# Save model
with open('models/baseline/xgb_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)
print("✓ Saved: models/baseline/xgb_regressor.pkl")

# -------- XGBoost Classification --------
print("\n[3.2] XGBoost - Classification Task")
print("-" * 80)

print("Training XGBoost Classifier...")
start_time = time.time()

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_cls_train == 0).sum() / (y_cls_train == 1).sum()

xgb_cls = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_cls.fit(X_train, y_cls_train)
train_time_xgb_cls = time.time() - start_time

print(f"✓ Training completed in {train_time_xgb_cls:.2f} seconds")

# Predictions
y_cls_pred_train_xgb = xgb_cls.predict(X_train)
y_cls_pred_test_xgb = xgb_cls.predict(X_test)
y_cls_proba_test_xgb = xgb_cls.predict_proba(X_test)[:, 1]

# Metrics
xgb_cls_metrics = {
    'model': 'XGBoost Classification',
    'train_accuracy': accuracy_score(y_cls_train, y_cls_pred_train_xgb),
    'test_accuracy': accuracy_score(y_cls_test, y_cls_pred_test_xgb),
    'train_precision': precision_score(y_cls_train, y_cls_pred_train_xgb, zero_division=0),
    'test_precision': precision_score(y_cls_test, y_cls_pred_test_xgb, zero_division=0),
    'train_recall': recall_score(y_cls_train, y_cls_pred_train_xgb, zero_division=0),
    'test_recall': recall_score(y_cls_test, y_cls_pred_test_xgb, zero_division=0),
    'train_f1': f1_score(y_cls_train, y_cls_pred_train_xgb, zero_division=0),
    'test_f1': f1_score(y_cls_test, y_cls_pred_test_xgb, zero_division=0),
    'test_auc': roc_auc_score(y_cls_test, y_cls_proba_test_xgb),
    'training_time': train_time_xgb_cls
}

print(f"\nClassification Metrics:")
print(f"  Train Accuracy:  {xgb_cls_metrics['train_accuracy']:.4f}")
print(f"  Test Accuracy:   {xgb_cls_metrics['test_accuracy']:.4f}")
print(f"  Test Precision:  {xgb_cls_metrics['test_precision']:.4f}")
print(f"  Test Recall:     {xgb_cls_metrics['test_recall']:.4f}")
print(f"  Test F1-Score:   {xgb_cls_metrics['test_f1']:.4f}")
print(f"  Test AUC-ROC:    {xgb_cls_metrics['test_auc']:.4f}")

print(f"\nConfusion Matrix:")
cm_xgb = confusion_matrix(y_cls_test, y_cls_pred_test_xgb)
print(cm_xgb)

results['xgb_classification'] = xgb_cls_metrics

# Save model
with open('models/baseline/xgb_classifier.pkl', 'wb') as f:
    pickle.dump(xgb_cls, f)
print("✓ Saved: models/baseline/xgb_classifier.pkl")

# ============================================================================
# STEP 4: Feature Importance Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 4] Feature Importance Analysis")
print("=" * 80)

# Get top 20 important features from XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features (XGBoost Regression):")
print(feature_importance.head(20).to_string(index=False))

# Save feature importance
feature_importance.to_csv('models/evaluation/feature_importance.csv', index=False)
print("\n✓ Saved: models/evaluation/feature_importance.csv")

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score')
plt.title('Top 20 Feature Importances (XGBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/plots/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/plots/feature_importance.png")
plt.close()

# ============================================================================
# STEP 5: Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Creating Visualizations")
print("=" * 80)

# Regression predictions plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Random Forest
axes[0].scatter(y_reg_test, y_reg_pred_test, alpha=0.5, s=50)
axes[0].plot([y_reg_test.min(), y_reg_test.max()], 
             [y_reg_test.min(), y_reg_test.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Sick %')
axes[0].set_ylabel('Predicted Sick %')
axes[0].set_title(f'Random Forest Regression\nTest R² = {rf_reg_metrics["test_r2"]:.4f}', 
                  fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# XGBoost
axes[1].scatter(y_reg_test, y_reg_pred_test_xgb, alpha=0.5, s=50, color='orange')
axes[1].plot([y_reg_test.min(), y_reg_test.max()], 
             [y_reg_test.min(), y_reg_test.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('Actual Sick %')
axes[1].set_ylabel('Predicted Sick %')
axes[1].set_title(f'XGBoost Regression\nTest R² = {xgb_reg_metrics["test_r2"]:.4f}', 
                  fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/plots/regression_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/plots/regression_predictions.png")
plt.close()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Random Forest
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Random Forest Classification\nF1 = {rf_cls_metrics["test_f1"]:.4f}', 
                  fontweight='bold')

# XGBoost
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title(f'XGBoost Classification\nF1 = {xgb_cls_metrics["test_f1"]:.4f}', 
                  fontweight='bold')

plt.tight_layout()
plt.savefig('models/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/plots/confusion_matrices.png")
plt.close()

# ROC Curves
plt.figure(figsize=(10, 8))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_cls_test, y_cls_proba_test)
plt.plot(fpr_rf, tpr_rf, linewidth=2, 
         label=f'Random Forest (AUC = {rf_cls_metrics["test_auc"]:.4f})')

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_cls_test, y_cls_proba_test_xgb)
plt.plot(fpr_xgb, tpr_xgb, linewidth=2,
         label=f'XGBoost (AUC = {xgb_cls_metrics["test_auc"]:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Classification Models', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('models/plots/roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/plots/roc_curves.png")
plt.close()

# ============================================================================
# STEP 6: Save Results Summary
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] Saving Results Summary")
print("=" * 80)

# Create results dataframe
results_df = pd.DataFrame(results).T
results_df.to_csv('models/evaluation/baseline_results.csv')
print("✓ Saved: models/evaluation/baseline_results.csv")

# Create detailed report
report = f"""
BASELINE MODELS EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

REGRESSION TASK - Predicting Sick Percentage
{'=' * 80}

Random Forest Regression:
  Train RMSE: {rf_reg_metrics['train_rmse']:.4f}
  Test RMSE:  {rf_reg_metrics['test_rmse']:.4f}
  Train MAE:  {rf_reg_metrics['train_mae']:.4f}
  Test MAE:   {rf_reg_metrics['test_mae']:.4f}
  Train R²:   {rf_reg_metrics['train_r2']:.4f}
  Test R²:    {rf_reg_metrics['test_r2']:.4f}
  Training Time: {rf_reg_metrics['training_time']:.2f}s

XGBoost Regression:
  Train RMSE: {xgb_reg_metrics['train_rmse']:.4f}
  Test RMSE:  {xgb_reg_metrics['test_rmse']:.4f}
  Train MAE:  {xgb_reg_metrics['train_mae']:.4f}
  Test MAE:   {xgb_reg_metrics['test_mae']:.4f}
  Train R²:   {xgb_reg_metrics['train_r2']:.4f}
  Test R²:    {xgb_reg_metrics['test_r2']:.4f}
  Training Time: {xgb_reg_metrics['training_time']:.2f}s

CLASSIFICATION TASK - Predicting High Risk Days
{'=' * 80}

Random Forest Classification:
  Train Accuracy: {rf_cls_metrics['train_accuracy']:.4f}
  Test Accuracy:  {rf_cls_metrics['test_accuracy']:.4f}
  Precision:      {rf_cls_metrics['test_precision']:.4f}
  Recall:         {rf_cls_metrics['test_recall']:.4f}
  F1-Score:       {rf_cls_metrics['test_f1']:.4f}
  AUC-ROC:        {rf_cls_metrics['test_auc']:.4f}
  Training Time:  {rf_cls_metrics['training_time']:.2f}s

XGBoost Classification:
  Train Accuracy: {xgb_cls_metrics['train_accuracy']:.4f}
  Test Accuracy:  {xgb_cls_metrics['test_accuracy']:.4f}
  Precision:      {xgb_cls_metrics['test_precision']:.4f}
  Recall:         {xgb_cls_metrics['test_recall']:.4f}
  F1-Score:       {xgb_cls_metrics['test_f1']:.4f}
  AUC-ROC:        {xgb_cls_metrics['test_auc']:.4f}
  Training Time:  {xgb_cls_metrics['training_time']:.2f}s

SUMMARY
{'=' * 80}
Best Regression Model: {"XGBoost" if xgb_reg_metrics['test_r2'] > rf_reg_metrics['test_r2'] else "Random Forest"}
Best Classification Model: {"XGBoost" if xgb_cls_metrics['test_f1'] > rf_cls_metrics['test_f1'] else "Random Forest"}

Files Saved:
  - models/baseline/rf_regressor.pkl
  - models/baseline/rf_classifier.pkl
  - models/baseline/xgb_regressor.pkl
  - models/baseline/xgb_classifier.pkl
  - models/evaluation/baseline_results.csv
  - models/evaluation/feature_importance.csv
  - models/plots/feature_importance.png
  - models/plots/regression_predictions.png
  - models/plots/confusion_matrices.png
  - models/plots/roc_curves.png
"""

with open('models/evaluation/baseline_report.txt', 'w') as f:
    f.write(report)

print(report)

print("\n" + "=" * 80)
print("✅ BASELINE MODELS TRAINING COMPLETE!")
print("=" * 80)
print("\nNext: Train PyTorch Neural Networks")