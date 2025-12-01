"""
Custom Federated Learning Implementation - Health MLOps Project
Federated Learning from scratch using PyTorch (Python 3.14 compatible)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List, Tuple
from collections import OrderedDict
import copy
import time
from datetime import datetime

print("=" * 80)
print("CUSTOM FEDERATED LEARNING - Health MLOps Project")
print("Python 3.14 Compatible Implementation")
print("=" * 80)

# ============================================================================
# 1. PYTORCH MODEL ARCHITECTURE
# ============================================================================

class HealthRiskNN(nn.Module):
    """Neural Network for Health Risk Prediction"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout=0.3):
        super(HealthRiskNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ============================================================================
# 2. FEDERATED CLIENT (Each Hospital/City Node)
# ============================================================================

class FederatedClient:
    """Client representing a single hospital/city node"""
    
    def __init__(self, client_id, X_train, y_train, X_test, y_test, 
                 input_size, task="regression", device='cpu'):
        self.client_id = client_id
        self.task = task
        self.device = device
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        self.X_test = torch.FloatTensor(X_test).to(device)
        self.y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)
        
        # Create DataLoaders
        train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        test_dataset = TensorDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Initialize local model
        output_size = 1 if task == "regression" else 2
        self.model = HealthRiskNN(input_size, output_size=output_size).to(device)
        
        # Loss and optimizer
        if task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"  ✓ Client {client_id}: {len(self.X_train)} train, {len(self.X_test)} test samples")
    
    def get_model_params(self):
        """Get current model parameters"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        """Set model parameters from server"""
        self.model.load_state_dict(params)
    
    def train_local(self, epochs=5):
        """Train model on local data"""
        self.model.train()
        
        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                
                if self.task == "classification":
                    batch_y = batch_y.squeeze().long()
                
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss / len(self.train_loader)
        
        avg_loss = total_loss / epochs
        return avg_loss
    
    def evaluate_local(self):
        """Evaluate model on local test data"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                outputs = self.model(batch_X)
                
                if self.task == "regression":
                    loss = self.criterion(outputs, batch_y)
                    total_loss += loss.item()
                else:
                    batch_y = batch_y.squeeze().long()
                    loss = self.criterion(outputs, batch_y)
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy, len(self.X_test)

# ============================================================================
# 3. FEDERATED SERVER
# ============================================================================

class FederatedServer:
    """Central server coordinating federated learning"""
    
    def __init__(self, input_size, task="regression", device='cpu'):
        self.task = task
        self.device = device
        
        # Initialize global model
        output_size = 1 if task == "regression" else 2
        self.global_model = HealthRiskNN(input_size, output_size=output_size).to(device)
        
        print(f"✓ Server initialized with global model")
    
    def get_global_params(self):
        """Get global model parameters"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def aggregate_params(self, client_params_list, client_weights):
        """
        FedAvg: Aggregate client parameters using weighted average
        
        Args:
            client_params_list: List of client model parameters
            client_weights: List of weights (typically data size) for each client
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated_params = OrderedDict()
        
        # Weighted average of all parameters
        for key in client_params_list[0].keys():
            aggregated_params[key] = sum(
                weights[i] * client_params_list[i][key]
                for i in range(len(client_params_list))
            )
        
        # Update global model
        self.global_model.load_state_dict(aggregated_params)
        
        return aggregated_params

# ============================================================================
# 4. FEDERATED LEARNING ORCHESTRATOR
# ============================================================================

class FederatedLearning:
    """Orchestrates the federated learning process"""
    
    def __init__(self, clients, server):
        self.clients = clients
        self.server = server
        self.history = {
            'round': [],
            'train_loss': [],
            'test_loss': [],
            'accuracy': []
        }
    
    def train_round(self, round_num, local_epochs=5):
        """Execute one round of federated learning"""
        
        print(f"\n{'='*70}")
        print(f"ROUND {round_num}")
        print(f"{'='*70}")
        
        # 1. Server broadcasts global model to all clients
        global_params = self.server.get_global_params()
        
        # 2. Each client trains locally
        print(f"\n[Phase 1] Local Training at {len(self.clients)} nodes...")
        client_params = []
        client_weights = []
        train_losses = []
        
        for client in self.clients:
            # Set global model parameters
            client.set_model_params(global_params)
            
            # Train locally
            train_loss = client.train_local(epochs=local_epochs)
            train_losses.append(train_loss)
            
            # Get updated parameters
            client_params.append(client.get_model_params())
            client_weights.append(len(client.X_train))
            
            print(f"  ✓ Client {client.client_id}: Train Loss = {train_loss:.4f}")
        
        # 3. Server aggregates client updates
        print(f"\n[Phase 2] Server aggregating updates from {len(self.clients)} clients...")
        self.server.aggregate_params(client_params, client_weights)
        print(f"  ✓ Global model updated")
        
        # 4. Evaluate global model on all clients
        print(f"\n[Phase 3] Evaluating global model...")
        global_params = self.server.get_global_params()
        
        test_losses = []
        accuracies = []
        total_samples = 0
        
        for client in self.clients:
            client.set_model_params(global_params)
            test_loss, accuracy, num_samples = client.evaluate_local()
            
            test_losses.append(test_loss * num_samples)
            accuracies.append(accuracy * num_samples)
            total_samples += num_samples
            
            print(f"  ✓ Client {client.client_id}: Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Weighted average of metrics
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = sum(test_losses) / total_samples
        avg_accuracy = sum(accuracies) / total_samples
        
        # Save history
        self.history['round'].append(round_num)
        self.history['train_loss'].append(avg_train_loss)
        self.history['test_loss'].append(avg_test_loss)
        self.history['accuracy'].append(avg_accuracy)
        
        print(f"\n{'='*70}")
        print(f"Round {round_num} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Avg Test Loss:  {avg_test_loss:.4f}")
        print(f"  Avg Accuracy:   {avg_accuracy:.4f}")
        print(f"{'='*70}")
        
        return avg_train_loss, avg_test_loss, avg_accuracy

# ============================================================================
# 5. MAIN TRAINING FUNCTION
# ============================================================================

def train_federated_model(task="regression", num_rounds=20, local_epochs=5):
    """
    Main federated learning training loop
    
    Args:
        task: "regression" or "classification"
        num_rounds: Number of federated rounds
        local_epochs: Epochs per local training
    """
    
    print(f"\n{'🔵'*40}")
    print(f"FEDERATED LEARNING: {task.upper()}")
    print(f"{'🔵'*40}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ========================================================================
    # Load data from all nodes
    # ========================================================================
    
    print(f"\n[STEP 1] Loading federated data from nodes...")
    print("-" * 80)
    
    clients_data = []
    input_size = None
    num_clients = 5
    
    for i in range(num_clients):
        node_file = f'data/federated/node_{i}_city_{i+1}.pkl'
        
        if not os.path.exists(node_file):
            print(f"❌ Node file not found: {node_file}")
            continue
        
        with open(node_file, 'rb') as f:
            node_data = pickle.load(f)
        
        # Get appropriate target - using correct key names
        if task == "regression":
            y_train = node_data['y_regression_train']
            y_test = node_data['y_regression_test']
        else:
            y_train = node_data['y_classification_train']
            y_test = node_data['y_classification_test']
        
        clients_data.append({
            'X_train': node_data['X_train'],
            'y_train': y_train,
            'X_test': node_data['X_test'],
            'y_test': y_test
        })
        
        if input_size is None:
            input_size = node_data['X_train'].shape[1]
    
    print(f"✓ Loaded {len(clients_data)} nodes with {input_size} features")
    
    # ========================================================================
    # Initialize clients and server
    # ========================================================================
    
    print(f"\n[STEP 2] Initializing federated learning system...")
    print("-" * 80)
    
    # Create clients
    clients = []
    for i, data in enumerate(clients_data):
        client = FederatedClient(
            client_id=i+1,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_test=data['X_test'],
            y_test=data['y_test'],
            input_size=input_size,
            task=task,
            device=device
        )
        clients.append(client)
    
    # Create server
    server = FederatedServer(input_size, task=task, device=device)
    
    # Create FL orchestrator
    fl_system = FederatedLearning(clients, server)
    
    print(f"\n✓ Federated Learning system ready!")
    print(f"  • Nodes: {len(clients)}")
    print(f"  • Rounds: {num_rounds}")
    print(f"  • Local epochs per round: {local_epochs}")
    
    # ========================================================================
    # Training loop
    # ========================================================================
    
    print(f"\n[STEP 3] Starting Federated Training")
    print("=" * 80)
    
    start_time = time.time()
    
    for round_num in range(1, num_rounds + 1):
        fl_system.train_round(round_num, local_epochs=local_epochs)
    
    training_time = time.time() - start_time
    
    # ========================================================================
    # Save results
    # ========================================================================
    
    print(f"\n[STEP 4] Saving results...")
    print("-" * 80)
    
    os.makedirs('models/federated', exist_ok=True)
    
    # Save model
    model_path = f'models/federated/{task}_global_model.pth'
    torch.save(server.global_model.state_dict(), model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Save history
    history_df = pd.DataFrame(fl_system.history)
    history_path = f'models/federated/{task}_training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"✓ History saved: {history_path}")
    
    # Save full results
    results = {
        'task': task,
        'num_rounds': num_rounds,
        'num_clients': len(clients),
        'local_epochs': local_epochs,
        'training_time': training_time,
        'history': fl_system.history,
        'final_train_loss': fl_system.history['train_loss'][-1],
        'final_test_loss': fl_system.history['test_loss'][-1],
        'final_accuracy': fl_system.history['accuracy'][-1]
    }
    
    results_path = f'models/federated/{task}_fl_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved: {results_path}")
    
    # ========================================================================
    # Final summary
    # ========================================================================
    
    print(f"\n{'='*80}")
    print(f"✅ FEDERATED LEARNING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTraining Summary:")
    print(f"  • Task: {task}")
    print(f"  • Total Rounds: {num_rounds}")
    print(f"  • Total Time: {training_time:.2f}s")
    print(f"  • Avg Time per Round: {training_time/num_rounds:.2f}s")
    print(f"\nFinal Performance:")
    print(f"  • Train Loss: {results['final_train_loss']:.4f}")
    print(f"  • Test Loss:  {results['final_test_loss']:.4f}")
    print(f"  • Accuracy:   {results['final_accuracy']:.4f}")
    print(f"{'='*80}")
    
    return results

# ============================================================================
# 6. RUN BOTH TASKS
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING TRAINING - Health MLOps Project")
    print("=" * 80)
    
    # Train Regression Model
    print("\n\n" + "🔵" * 40)
    print("TASK 1: FEDERATED REGRESSION (Predicting Sick Percentage)")
    print("🔵" * 40)
    
    reg_results = train_federated_model(
        task="regression",
        num_rounds=20,
        local_epochs=5
    )
    
    # Train Classification Model
    print("\n\n" + "🟢" * 40)
    print("TASK 2: FEDERATED CLASSIFICATION (Predicting High Risk)")
    print("🟢" * 40)
    
    clf_results = train_federated_model(
        task="classification",
        num_rounds=20,
        local_epochs=5
    )
    
    # ========================================================================
    # Overall Summary
    # ========================================================================
    
    print("\n\n" + "=" * 80)
    print("ALL FEDERATED LEARNING TRAINING COMPLETE!")
    print("=" * 80)
    
    print("\nFiles Created:")
    print("  Models:")
    print("    - models/federated/regression_global_model.pth")
    print("    - models/federated/classification_global_model.pth")
    print("\n  Training History:")
    print("    - models/federated/regression_training_history.csv")
    print("    - models/federated/classification_training_history.csv")
    print("\n  Results:")
    print("    - models/federated/regression_fl_results.pkl")
    print("    - models/federated/classification_fl_results.pkl")
    
    print("\nWhat Was Accomplished:")
    print("  ✅ Trained models across 5 distributed hospital/city nodes")
    print("  ✅ Data remained private at each node (never centralized)")
    print("  ✅ Only model parameters were shared between nodes")
    print("  ✅ Achieved collaborative learning without privacy compromise")
    print("  ✅ Used FedAvg algorithm for parameter aggregation")
    
    print("\nComparison with Centralized Baseline:")
    print(f"  Regression - FL Test Loss: {reg_results['final_test_loss']:.4f}")
    print(f"  Classification - FL Accuracy: {clf_results['final_accuracy']:.4f}")
    
    print("\nNext Steps:")
    print("  1. ✅ Create comprehensive model comparison report")
    print("  2. ⏭️  Build monitoring dashboard")
    print("  3. ⏭️  Deploy with Docker/Kubernetes")
    print("  4. ⏭️  Set up CI/CD pipeline")
    
    print("=" * 80)