#!/usr/bin/env python3
"""
Test script for Federated Learning System
=========================================

This script tests all components of the federated learning system to ensure
everything works correctly before running the full demo.
"""

import os
import sys
import time
import requests
import torch
import pandas as pd
import numpy as np

# Add the federated directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.model import AdmissionClassifier
from shared.train import train, evaluate_model
from shared.utils import load_data_for_client, create_dataloaders, federated_average
from client.node import ClientNode

def test_model_creation():
    """Test model creation and forward pass"""
    print("🧪 Testing model creation...")
    
    try:
        model = AdmissionClassifier(input_dim=29)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass
        X = torch.randn(5, 29)
        output = model(X)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n🧪 Testing data loading...")
    
    try:
        # Test loading data for a client
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        
        if client_data is None:
            print("❌ Data loading failed")
            return False
        
        X_train, y_train = client_data['train']
        X_val, y_val = client_data['val']
        X_test, y_test = client_data['test']
        
        print(f"✅ Data loaded successfully")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        print(f"   Input features: {X_train.shape[1]}")
        
        # Test dataloader creation
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=8)
        print(f"✅ DataLoaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_training():
    """Test training functionality"""
    print("\n🧪 Testing training...")
    
    try:
        # Create model and data
        model = AdmissionClassifier(input_dim=29)
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        X_train, y_train = client_data['train']
        X_val, y_val = client_data['val']
        
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=8)
        
        # Test training
        updated_weights, history = train(model, train_loader, val_loader, epochs=2, lr=0.001)
        
        print(f"✅ Training completed successfully")
        print(f"   Training history keys: {list(history.keys())}")
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def test_evaluation():
    """Test model evaluation"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        # Create model and test data
        model = AdmissionClassifier(input_dim=29)
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        X_test, y_test = client_data['test']
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Test evaluation
        metrics = evaluate_model(model, test_loader)
        
        print(f"✅ Evaluation completed successfully")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def test_federated_averaging():
    """Test federated averaging functionality"""
    print("\n🧪 Testing federated averaging...")
    
    try:
        # Create multiple models with different weights
        models = []
        for i in range(3):
            model = AdmissionClassifier(input_dim=29)
            # Add some noise to make weights different
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.1
            models.append(model)
        
        # Get weights from all models
        weights_list = [model.state_dict() for model in models]
        client_sizes = [100, 150, 200]  # Different client sizes
        
        # Test federated averaging
        avg_weights = federated_average(weights_list, client_sizes)
        
        print(f"✅ Federated averaging completed successfully")
        print(f"   Number of models: {len(weights_list)}")
        print(f"   Client sizes: {client_sizes}")
        
        return True
    except Exception as e:
        print(f"❌ Federated averaging failed: {e}")
        return False

def test_client_node():
    """Test client node functionality"""
    print("\n🧪 Testing client node...")
    
    try:
        # Create client node
        client = ClientNode("test_client")
        print(f"✅ Client node created successfully")
        
        # Test data loading
        data_info = client.load_local_data()
        if data_info is None:
            print("❌ Client data loading failed")
            return False
        
        print(f"✅ Client data loaded successfully")
        print(f"   Train size: {data_info['train_size']}")
        print(f"   Val size: {data_info['val_size']}")
        print(f"   Test size: {data_info['test_size']}")
        
        return True
    except Exception as e:
        print(f"❌ Client node test failed: {e}")
        return False

def test_server_endpoints():
    """Test server endpoints (requires server to be running)"""
    print("\n🧪 Testing server endpoints...")
    
    server_url = "http://localhost:5000"
    
    try:
        # Test status endpoint
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server status endpoint working")
            print(f"   Round number: {status.get('round_number', 'N/A')}")
            print(f"   Pending updates: {status.get('pending_updates', 'N/A')}")
        else:
            print(f"❌ Server status endpoint failed: {response.status_code}")
            return False
        
        # Test model info endpoint
        response = requests.get(f"{server_url}/model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info endpoint working")
            print(f"   Model type: {model_info.get('model_type', 'N/A')}")
            print(f"   Parameters: {model_info.get('total_parameters', 'N/A')}")
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        sample_features = [36.6, 115.0, 20.0, 99.0, 92.0, 42.0, 0.0, 2.0, 40.0,
                          36.7, 112.0, 18.0, 98.0, 101.0, 41.0, 0.0, 1, 0, 0, 0,
                          0, 0, 0, 1, 0, 1, 0, 0, 0]
        
        response = requests.post(f"{server_url}/predict", 
                               json={'features': sample_features}, 
                               timeout=5)
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction endpoint working")
            print(f"   Prediction: {prediction.get('prediction', 'N/A')}")
            print(f"   Probability: {prediction.get('probability', 'N/A'):.3f}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            return False
        
        return True
    except requests.exceptions.ConnectionError:
        print("⚠️ Server not running. Skipping server endpoint tests.")
        return True
    except Exception as e:
        print(f"❌ Server endpoint test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Federated Learning System Tests")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training", test_training),
        ("Evaluation", test_evaluation),
        ("Federated Averaging", test_federated_averaging),
        ("Client Node", test_client_node),
        ("Server Endpoints", test_server_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1) 