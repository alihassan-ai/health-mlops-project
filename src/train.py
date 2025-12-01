"""
PyTorch Neural Networks Training - Health MLOps Project
Deep Learning models for Regression and Classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
import time
from datetime import datetime
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt

print("=" * 80)
print("PYTORCH NEURAL NETWORKS TRAINING - Health MLOps Project")
print("=" * 80)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🔥 Using device: {device}")

# Create directories
os.makedirs('models/pytorch', exist_ok=True)
os.makedirs('models/pytorch_plots', exist_ok=True)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[STEP 1] Loading data...")
print("-" * 80)

with open('data/processed/centralized_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train_scaled']
X_test = data['X_test_scaled']
y_reg_train = data['y_regression_train']
y_reg_test = data['y_regression_test']
y_cls_train = data['y_classification_train']
y_cls_test = data['y_classification_test']

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Input features: {X_train.shape[1]}")

# ============================================================================
# STEP 2: Create PyTorch Datasets
# ============================================================================
print("\n[STEP 2] Creating PyTorch datasets...")
print("-" * 80)

class HealthDataset(Dataset):
    """Custom Dataset for health data"""
    def __init__(self, X, y, task='regression'):
        self.X = torch.FloatTensor(X)
        if task == 'regression':
            self.y = torch.FloatTensor(y)
        else:  # classification
            self.y = torch.LongTensor(y)
        self.task = task
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
train_dataset_reg = HealthDataset(X_train, y_reg_train, task='regression')
test_dataset_reg = HealthDataset(X_test, y_reg_test, task='regression')

train_dataset_cls = HealthDataset(X_train, y_cls_train, task='classification')
test_dataset_cls = HealthDataset(X_test, y_cls_test, task='classification')

# Create dataloaders
batch_size = 32

train_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=True)
test_loader_reg = DataLoader(test_dataset_reg, batch_size=batch_size, shuffle=False)

train_loader_cls = DataLoader(train_dataset_cls, batch_size=batch_size, shuffle=True)
test_loader_cls = DataLoader(test_dataset_cls, batch_size=batch_size, shuffle=False)

print(f"✓ Created dataloaders with batch size: {batch_size}")

# ============================================================================
# STEP 3: Define Neural Network Architectures
# ============================================================================
print("\n[STEP 3] Defining neural network architectures...")
print("-" * 80)

class RegressionNN(nn.Module):
    """Neural Network for Regression"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(RegressionNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ClassificationNN(nn.Module):
    """Neural Network for Binary Classification"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout=0.3):
        super(ClassificationNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 2))  # 2 classes
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

input_size = X_train.shape[1]
print(f"✓ Input size: {input_size}")
print(f"✓ Architecture: [{input_size}] -> [128, 64, 32] -> [output]")

# ============================================================================
# STEP 4: Training Functions
# ============================================================================

def train_regression_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    """Train regression model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print("\nTraining Regression Model...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/pytorch/best_regression_nn.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f}s")
    
    # Load best model
    model.load_state_dict(torch.load('models/pytorch/best_regression_nn.pth'))
    
    return model, train_losses, test_losses, training_time

def train_classification_model(model, train_loader, test_loader, epochs=100, lr=0.001):
    """Train classification model"""
    # Weighted loss for imbalanced classes
    class_counts = np.bincount(y_cls_train)
    class_weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=10)
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print("\nTraining Classification Model...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/pytorch/best_classification_nn.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f}s")
    
    # Load best model
    model.load_state_dict(torch.load('models/pytorch/best_classification_nn.pth'))
    
    return model, train_losses, test_losses, training_time

# ============================================================================
# STEP 5: Train Regression Model
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Training Regression Neural Network")
print("=" * 80)

reg_model = RegressionNN(input_size).to(device)
print(f"\n✓ Model created with {sum(p.numel() for p in reg_model.parameters())} parameters")

reg_model, train_losses_reg, test_losses_reg, train_time_reg = train_regression_model(
    reg_model, train_loader_reg, test_loader_reg, epochs=100, lr=0.001
)

# Evaluate regression model
print("\nEvaluating Regression Model...")
reg_model.eval()
with torch.no_grad():
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    y_train_pred = reg_model(X_train_tensor).squeeze().cpu().numpy()
    y_test_pred = reg_model(X_test_tensor).squeeze().cpu().numpy()

reg_metrics = {
    'model': 'PyTorch NN Regression',
    'train_rmse': np.sqrt(mean_squared_error(y_reg_train, y_train_pred)),
    'test_rmse': np.sqrt(mean_squared_error(y_reg_test, y_test_pred)),
    'train_mae': mean_absolute_error(y_reg_train, y_train_pred),
    'test_mae': mean_absolute_error(y_reg_test, y_test_pred),
    'train_r2': r2_score(y_reg_train, y_train_pred),
    'test_r2': r2_score(y_reg_test, y_test_pred),
    'training_time': train_time_reg
}

print(f"\nRegression Metrics:")
print(f"  Train RMSE: {reg_metrics['train_rmse']:.4f}")
print(f"  Test RMSE:  {reg_metrics['test_rmse']:.4f}")
print(f"  Train MAE:  {reg_metrics['train_mae']:.4f}")
print(f"  Test MAE:   {reg_metrics['test_mae']:.4f}")
print(f"  Train R²:   {reg_metrics['train_r2']:.4f}")
print(f"  Test R²:    {reg_metrics['test_r2']:.4f}")

# ============================================================================
# STEP 6: Train Classification Model
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 6] Training Classification Neural Network")
print("=" * 80)

cls_model = ClassificationNN(input_size).to(device)
print(f"\n✓ Model created with {sum(p.numel() for p in cls_model.parameters())} parameters")

cls_model, train_losses_cls, test_losses_cls, train_time_cls = train_classification_model(
    cls_model, train_loader_cls, test_loader_cls, epochs=100, lr=0.001
)

# Evaluate classification model
print("\nEvaluating Classification Model...")
cls_model.eval()
with torch.no_grad():
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    train_outputs = cls_model(X_train_tensor)
    test_outputs = cls_model(X_test_tensor)
    
    _, y_train_pred_cls = torch.max(train_outputs, 1)
    _, y_test_pred_cls = torch.max(test_outputs, 1)
    
    y_train_pred_cls = y_train_pred_cls.cpu().numpy()
    y_test_pred_cls = y_test_pred_cls.cpu().numpy()
    
    # Get probabilities
    y_test_proba = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

cls_metrics = {
    'model': 'PyTorch NN Classification',
    'train_accuracy': accuracy_score(y_cls_train, y_train_pred_cls),
    'test_accuracy': accuracy_score(y_cls_test, y_test_pred_cls),
    'test_precision': precision_score(y_cls_test, y_test_pred_cls, zero_division=0),
    'test_recall': recall_score(y_cls_test, y_test_pred_cls, zero_division=0),
    'test_f1': f1_score(y_cls_test, y_test_pred_cls, zero_division=0),
    'test_auc': roc_auc_score(y_cls_test, y_test_proba) if len(np.unique(y_cls_test)) > 1 else 0.0,
    'training_time': train_time_cls
}

print(f"\nClassification Metrics:")
print(f"  Train Accuracy: {cls_metrics['train_accuracy']:.4f}")
print(f"  Test Accuracy:  {cls_metrics['test_accuracy']:.4f}")
print(f"  Test Precision: {cls_metrics['test_precision']:.4f}")
print(f"  Test Recall:    {cls_metrics['test_recall']:.4f}")
print(f"  Test F1-Score:  {cls_metrics['test_f1']:.4f}")
print(f"  Test AUC-ROC:   {cls_metrics['test_auc']:.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_cls_test, y_test_pred_cls)
print(cm)

# ============================================================================
# STEP 7: Visualizations
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 7] Creating Visualizations")
print("=" * 80)

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Regression loss
axes[0].plot(train_losses_reg, label='Train Loss', linewidth=2)
axes[0].plot(test_losses_reg, label='Test Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Regression Model - Training Curves', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Classification loss
axes[1].plot(train_losses_cls, label='Train Loss', linewidth=2)
axes[1].plot(test_losses_cls, label='Test Loss', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Cross-Entropy Loss')
axes[1].set_title('Classification Model - Training Curves', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('models/pytorch_plots/training_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/pytorch_plots/training_curves.png")
plt.close()

# Regression predictions
plt.figure(figsize=(10, 8))
plt.scatter(y_reg_test, y_test_pred, alpha=0.5, s=50)
plt.plot([y_reg_test.min(), y_reg_test.max()], 
         [y_reg_test.min(), y_reg_test.max()], 
         'r--', lw=2, label='Perfect prediction')
plt.xlabel('Actual Sick %')
plt.ylabel('Predicted Sick %')
plt.title(f'PyTorch NN Regression\nTest R² = {reg_metrics["test_r2"]:.4f}', 
          fontweight='bold', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('models/pytorch_plots/regression_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: models/pytorch_plots/regression_predictions.png")
plt.close()

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Saving Results")
print("=" * 80)

# Save metrics
results = {
    'pytorch_regression': reg_metrics,
    'pytorch_classification': cls_metrics
}

results_df = pd.DataFrame(results).T
results_df.to_csv('models/evaluation/pytorch_results.csv')
print("✓ Saved: models/evaluation/pytorch_results.csv")

# Create report
report = f"""
PYTORCH NEURAL NETWORKS EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Device: {device}
{'=' * 80}

REGRESSION TASK - Predicting Sick Percentage
{'=' * 80}
Architecture: [{input_size}] -> [128, 64, 32] -> [1]
Parameters: {sum(p.numel() for p in reg_model.parameters())}

Metrics:
  Train RMSE: {reg_metrics['train_rmse']:.4f}
  Test RMSE:  {reg_metrics['test_rmse']:.4f}
  Train MAE:  {reg_metrics['train_mae']:.4f}
  Test MAE:   {reg_metrics['test_mae']:.4f}
  Train R²:   {reg_metrics['train_r2']:.4f}
  Test R²:    {reg_metrics['test_r2']:.4f}
  Training Time: {reg_metrics['training_time']:.2f}s

CLASSIFICATION TASK - Predicting High Risk Days
{'=' * 80}
Architecture: [{input_size}] -> [128, 64, 32] -> [2]
Parameters: {sum(p.numel() for p in cls_model.parameters())}

Metrics:
  Train Accuracy: {cls_metrics['train_accuracy']:.4f}
  Test Accuracy:  {cls_metrics['test_accuracy']:.4f}
  Precision:      {cls_metrics['test_precision']:.4f}
  Recall:         {cls_metrics['test_recall']:.4f}
  F1-Score:       {cls_metrics['test_f1']:.4f}
  AUC-ROC:        {cls_metrics['test_auc']:.4f}
  Training Time:  {cls_metrics['training_time']:.2f}s

FILES SAVED
{'=' * 80}
  - models/pytorch/best_regression_nn.pth
  - models/pytorch/best_classification_nn.pth
  - models/evaluation/pytorch_results.csv
  - models/pytorch_plots/training_curves.png
  - models/pytorch_plots/regression_predictions.png
"""

with open('models/evaluation/pytorch_report.txt', 'w') as f:
    f.write(report)

print(report)

print("\n" + "=" * 80)
print("✅ PYTORCH NEURAL NETWORKS TRAINING COMPLETE!")
print("=" * 80)
print("\nNext: Implement Federated Learning with Flower")