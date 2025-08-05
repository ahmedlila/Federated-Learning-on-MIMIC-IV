#!/usr/bin/env python3
"""
Complete End-to-End Federated Learning Demo
Shows the entire workflow from start to finish
"""

import sys
import os
import time
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add federated module to path
sys.path.append('src/federated')

from shared.model import AdmissionClassifier
from shared.utils import get_model_weights, set_model_weights, federated_average
from shared.train import train, evaluate_model
from client.node import ClientNode
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("🏥 FEDERATED LEARNING FOR HOSPITAL ADMISSION PREDICTION")
print("Complete End-to-End Demonstration")
print("=" * 80)
print()
time.sleep(1)

# Track metrics
round_metrics = {
    'round': [],
    'client_losses': [],
    'global_accuracy': [],
    'global_loss': []
}

print("📋 DEMO OVERVIEW:")
print("-" * 60)
print("• Predict hospital admissions using MIMIC-IV ED data")
print("• 3 hospitals collaborate without sharing patient data")
print("• Each hospital trains on local data")
print("• Central server aggregates model updates")
print("• Privacy preserved throughout!")
print()
input("Press Enter to start the demonstration...")
print()

# PHASE 1: SETUP
print("🔧 PHASE 1: SYSTEM SETUP")
print("=" * 60)
print()

print("📡 1.1 Initializing Central Server")
print("-" * 40)
server_model = AdmissionClassifier(input_dim=29)
total_params = sum(p.numel() for p in server_model.parameters())
print(f"✅ Server initialized")
print(f"   • Model: Neural Network")
print(f"   • Architecture: 29 → 128 → 64 → 32 → 1")
print(f"   • Total parameters: {total_params:,}")
print()
time.sleep(1)

print("🏥 1.2 Creating Hospital Clients")
print("-" * 40)
hospital_names = ["St. Mary's Hospital", "City General Hospital", "Regional Medical Center"]
clients = []
for i, name in enumerate(hospital_names):
    client = ClientNode(f"client_{i+1}")
    clients.append(client)
    print(f"✅ {name} joined the federation")
    time.sleep(0.5)
print()

print("📊 1.3 Loading Data at Each Hospital")
print("-" * 40)
client_data = []
total_samples = 0
for i, (client, name) in enumerate(zip(clients, hospital_names)):
    data = client.load_local_data()
    if data:
        client_data.append(data)
        total_samples += data['train_size']
        print(f"✅ {name}:")
        print(f"   • Training samples: {data['train_size']}")
        print(f"   • Validation samples: {data['val_size']}")
        print(f"   • Test samples: {data['test_size']}")
        time.sleep(0.5)
print(f"\n📈 Total training samples across all hospitals: {total_samples}")
print()

# PHASE 2: FEDERATED TRAINING
n_rounds = 5
print(f"🚀 PHASE 2: FEDERATED LEARNING ({n_rounds} ROUNDS)")
print("=" * 60)
print()

# Load test data for global evaluation
X_test = pd.read_csv("data/processed/data_preprocessing/X_test.csv")
y_test = pd.read_csv("data/processed/data_preprocessing/y_test.csv")
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for round_num in range(n_rounds):
    print(f"📍 ROUND {round_num + 1}/{n_rounds}")
    print("-" * 40)
    
    # Step 1: Distribute global model
    print("📥 Step 1: Broadcasting global model to all hospitals...")
    for client, name in zip(clients, hospital_names):
        set_model_weights(client.model, get_model_weights(server_model))
    print("✅ All hospitals received the latest global model")
    print()
    time.sleep(0.5)
    
    # Step 2: Local training
    print("🏃 Step 2: Local training at each hospital...")
    client_weights = []
    client_sizes = []
    round_client_losses = []
    
    for i, (client, data, name) in enumerate(zip(clients, client_data, hospital_names)):
        print(f"\n   Training at {name}...")
        
        # Train for 3 epochs
        updated_weights, history = train(
            client.model,
            data['train_loader'],
            data['val_loader'],
            epochs=3,
            lr=0.001,
            device='cpu'
        )
        
        if updated_weights:
            client_weights.append(updated_weights)
            client_sizes.append(data['train_size'])
            
            # Get final losses
            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1] if 'val_loss' in history else final_train_loss
            round_client_losses.append(final_train_loss)
            
            print(f"   ✅ Training complete")
            print(f"      • Final train loss: {final_train_loss:.4f}")
            print(f"      • Final validation loss: {final_val_loss:.4f}")
            print(f"      • Final accuracy: {history['train_acc'][-1]:.2%}")
        
        time.sleep(0.5)
    
    # Step 3: Aggregate at server
    print("\n⚖️  Step 3: Server aggregating hospital updates...")
    print(f"   • Received updates from {len(client_weights)} hospitals")
    print(f"   • Hospital data sizes: {client_sizes}")
    
    averaged_weights = federated_average(client_weights, client_sizes)
    set_model_weights(server_model, averaged_weights)
    print("✅ Global model updated with aggregated knowledge")
    print()
    time.sleep(0.5)
    
    # Step 4: Evaluate global model
    print("📊 Step 4: Evaluating global model performance...")
    metrics = evaluate_model(server_model, test_loader, device='cpu')
    
    print(f"   📈 Global Model Performance:")
    print(f"      • Accuracy: {metrics['accuracy']:.2%}")
    print(f"      • Precision: {metrics['precision']:.2%}")
    print(f"      • Recall: {metrics['recall']:.2%}")
    print(f"      • F1-Score: {metrics['f1_score']:.3f}")
    
    # Store metrics
    round_metrics['round'].append(round_num + 1)
    round_metrics['client_losses'].append(np.mean(round_client_losses))
    round_metrics['global_accuracy'].append(metrics['accuracy'])
    
    print()
    print("=" * 60)
    print()
    time.sleep(1)

# PHASE 3: RESULTS
print("📈 PHASE 3: FINAL RESULTS")
print("=" * 60)
print()

print("🏆 Training Summary:")
print("-" * 40)
print(f"• Completed {n_rounds} federated rounds")
print(f"• {len(clients)} hospitals participated")
print(f"• Total samples used: {total_samples}")
print(f"• Final global accuracy: {round_metrics['global_accuracy'][-1]:.2%}")
print()

print("📊 Performance Progression:")
print("-" * 40)
for i, (round_num, acc) in enumerate(zip(round_metrics['round'], round_metrics['global_accuracy'])):
    improvement = "" if i == 0 else f" (+{(acc - round_metrics['global_accuracy'][i-1])*100:.1f}%)" if acc > round_metrics['global_accuracy'][i-1] else f" ({(acc - round_metrics['global_accuracy'][i-1])*100:.1f}%)"
    print(f"Round {round_num}: {acc:.2%}{improvement}")
print()

# Make some predictions
print("🔮 Sample Predictions on Test Data:")
print("-" * 40)
server_model.eval()
with torch.no_grad():
    # Get 5 random samples
    indices = np.random.choice(len(X_test), 5, replace=False)
    for idx in indices:
        sample = X_test_tensor[idx].unsqueeze(0)
        actual = int(y_test_tensor[idx].item())
        
        output = server_model(sample)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob > 0.5 else 0
        
        print(f"Patient {idx+1}:")
        print(f"  • Prediction: {'Admission' if pred == 1 else 'No Admission'} (confidence: {prob:.2%})")
        print(f"  • Actual: {'Admission' if actual == 1 else 'No Admission'}")
        print(f"  • {'✅ Correct' if pred == actual else '❌ Incorrect'}")
        print()

print("🔐 Privacy Summary:")
print("-" * 40)
print("✅ No patient data was shared between hospitals")
print("✅ Each hospital maintained complete data control")
print("✅ Only model parameters were exchanged")
print("✅ Patient privacy preserved throughout!")
print()

print("=" * 80)
print("🎉 FEDERATED LEARNING DEMONSTRATION COMPLETE!")
print("=" * 80)
print()
print("This demonstration showed how multiple hospitals can collaborate")
print("to build better AI models while preserving patient privacy.")
print("Federated learning enables the future of collaborative healthcare AI!")
print()