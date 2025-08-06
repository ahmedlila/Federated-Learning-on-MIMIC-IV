import sys
import os
# Add the federated directory to the path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pickle
import requests
import pandas as pd
import numpy as np
import json
import time
from torch.utils.data import DataLoader, TensorDataset
from shared.model import AdmissionClassifier
from shared.train import train, evaluate_model
from shared.utils import set_model_weights, get_model_weights, load_data_for_client, create_dataloaders

class ClientNode:
    def __init__(self, client_id, server_url='http://localhost:8080', device='cpu'):
        self.client_id = client_id
        self.server_url = server_url
        self.device = device
        self.model = AdmissionClassifier(input_dim=29).to(self.device)
        self.training_history = []
        
        print(f"[Client {client_id}] Initialized with device: {device}")
        print(f"[Client {client_id}] Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def fetch_global_weights(self):
        """Fetch global model weights from the server"""
        try:
            print(f"[Client {self.client_id}] Fetching global model weights...")
            headers = {
                'Accept': 'application/octet-stream'
            }
            response = requests.get(f"{self.server_url}/weights", headers=headers, timeout=10)
            if response.status_code == 200:
                weights = pickle.loads(response.content)
                set_model_weights(self.model, weights)
                print(f"[Client {self.client_id}] Global weights loaded successfully.")
                return True
            else:
                print(f"[Client {self.client_id}] Failed to fetch weights. Status: {response.status_code}")
                print(f"[Client {self.client_id}] Response content: {response.text[:200]}")
                return False
        except Exception as e:
            print(f"[Client {self.client_id}] Error fetching weights: {e}")
            return False

    def load_local_data(self, data_dir="../../data/processed/data_preprocessing"):
        """Load and prepare local data for this client"""
        try:
            print(f"[Client {self.client_id}] Loading local data...")
            
            # Check if non-IID data path is available
            if hasattr(self, 'non_iid_data_path') and self.non_iid_data_path:
                print(f"[Client {self.client_id}] Using non-IID data from: {self.non_iid_data_path}")
                
                # Load non-IID data directly
                X_train = pd.read_csv(f"{self.non_iid_data_path}/X_train.csv")
                y_train = pd.read_csv(f"{self.non_iid_data_path}/y_train.csv")
                X_val = pd.read_csv(f"{self.non_iid_data_path}/X_val.csv")
                y_val = pd.read_csv(f"{self.non_iid_data_path}/y_val.csv")
                X_test = pd.read_csv(f"{self.non_iid_data_path}/X_test.csv")
                y_test = pd.read_csv(f"{self.non_iid_data_path}/y_test.csv")
                
                # Convert to tensors
                X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze()
                X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).squeeze()
                X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
                y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()
                
                print(f"[Client {self.client_id}] Non-IID data loaded - Train: {X_train_tensor.shape[0]}, Val: {X_val_tensor.shape[0]}, Test: {X_test_tensor.shape[0]}")
                
            else:
                # Load data using the original method (IID)
                client_data = load_data_for_client(self.client_id, data_dir)
                
                X_train_tensor, y_train_tensor = client_data['train']
                X_val_tensor, y_val_tensor = client_data['val']
                X_test_tensor, y_test_tensor = client_data['test']
                
                print(f"[Client {self.client_id}] IID data loaded - Train: {X_train_tensor.shape[0]}, Val: {X_val_tensor.shape[0]}, Test: {X_test_tensor.shape[0]}")
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, batch_size=16, shuffle=True
            )
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'X_test': X_test_tensor,
                'y_test': y_test_tensor,
                'train_size': X_train_tensor.shape[0],
                'val_size': X_val_tensor.shape[0],
                'test_size': X_test_tensor.shape[0]
            }
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading data: {e}")
            return None

    def local_train(self, train_loader, val_loader=None, epochs=5, lr=0.001):
        """Perform local training on client data"""
        try:
            print(f"[Client {self.client_id}] Starting local training...")
            
            # Train the model
            updated_weights, history = train(
                self.model, 
                train_loader, 
                val_loader, 
                device=self.device, 
                epochs=epochs, 
                lr=lr
            )
            
            # Store training history
            self.training_history.append({
                'client_id': self.client_id,
                'epochs': epochs,
                'learning_rate': lr,
                'history': history,
                'timestamp': time.time()
            })
            
            print(f"[Client {self.client_id}] Local training completed.")
            return updated_weights
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error during training: {e}")
            return None

    def evaluate_local_model(self, test_loader):
        """Evaluate the local model"""
        try:
            metrics = evaluate_model(self.model, test_loader, device=self.device)
            print(f"[Client {self.client_id}] Local evaluation - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            return metrics
        except Exception as e:
            print(f"[Client {self.client_id}] Error evaluating model: {e}")
            return None

    def send_updated_weights(self, weights, client_size):
        """Send updated weights to the central server"""
        try:
            print(f"[Client {self.client_id}] Sending updated weights to server...")
            
            # Prepare the payload
            weights_bytes = pickle.dumps(weights)
            payload = {
                'weights': weights_bytes.decode('latin1'),
                'client_size': client_size,
                'client_id': self.client_id
            }
            
            response = requests.post(f"{self.server_url}/update", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                print(f"[Client {self.client_id}] Update sent successfully. Status: {result.get('status', 'unknown')}")
                return result
            else:
                print(f"[Client {self.client_id}] Failed to send update. Status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[Client {self.client_id}] Error sending weights: {e}")
            return None

    def predict(self, features):
        """Make a prediction using the local model"""
        try:
            self.model.eval()
            with torch.no_grad():
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                outputs = self.model(X)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).int()
            
            return {
                'prediction': int(predictions.item()),
                'probability': float(probabilities.item())
            }
        except Exception as e:
            print(f"[Client {self.client_id}] Error making prediction: {e}")
            return None

    def run_federated_round(self, epochs=5, lr=0.001):
        """Run a complete federated learning round"""
        print(f"[Client {self.client_id}] Starting federated learning round...")
        
        # Step 1: Load local data
        data_info = self.load_local_data()
        if data_info is None:
            print(f"[Client {self.client_id}] Failed to load data. Aborting round.")
            return False
        
        # Step 2: Fetch global model
        if not self.fetch_global_weights():
            print(f"[Client {self.client_id}] Failed to fetch global weights. Aborting round.")
            return False
        
        # Step 3: Evaluate model before training (LOCAL TESTING)
        print(f"[Client {self.client_id}] Testing model before training...")
        test_dataset = TensorDataset(data_info['X_test'], data_info['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        pre_train_metrics = self.evaluate_local_model(test_loader)
        
        if pre_train_metrics:
            print(f"[Client {self.client_id}] Pre-training test results - Accuracy: {pre_train_metrics['accuracy']:.4f}, F1: {pre_train_metrics['f1_score']:.4f}")
        
        # Step 4: Local training
        print(f"[Client {self.client_id}] Starting local training...")
        updated_weights = self.local_train(
            data_info['train_loader'], 
            data_info['val_loader'], 
            epochs=epochs, 
            lr=lr
        )
        
        if updated_weights is None:
            print(f"[Client {self.client_id}] Training failed. Aborting round.")
            return False
        
        # Step 5: Evaluate model after training (LOCAL TESTING)
        print(f"[Client {self.client_id}] Testing model after training...")
        post_train_metrics = self.evaluate_local_model(test_loader)
        
        if post_train_metrics:
            print(f"[Client {self.client_id}] Post-training test results - Accuracy: {post_train_metrics['accuracy']:.4f}, F1: {post_train_metrics['f1_score']:.4f}")
            
            # Calculate improvement
            if pre_train_metrics:
                acc_improvement = post_train_metrics['accuracy'] - pre_train_metrics['accuracy']
                f1_improvement = post_train_metrics['f1_score'] - pre_train_metrics['f1_score']
                print(f"[Client {self.client_id}] Local improvement - Accuracy: {acc_improvement:+.4f}, F1: {f1_improvement:+.4f}")
        
        # Step 6: Send updated weights to server
        result = self.send_updated_weights(updated_weights, data_info['train_size'])
        
        if result:
            print(f"[Client {self.client_id}] Federated round completed successfully.")
            
            # Store testing results
            round_results = {
                'client_id': self.client_id,
                'pre_training_metrics': pre_train_metrics,
                'post_training_metrics': post_train_metrics,
                'train_size': data_info['train_size'],
                'test_size': data_info['test_size'],
                'timestamp': time.time()
            }
            
            # Add to training history
            self.training_history.append(round_results)
            
            return True
        else:
            print(f"[Client {self.client_id}] Federated round failed.")
            return False

    def get_server_status(self):
        """Get current server status"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            response = requests.get(f"{self.server_url}/status", headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Client {self.client_id}] Failed to get server status. Status: {response.status_code}")
                print(f"[Client {self.client_id}] Response content: {response.text[:200]}")
                return None
        except Exception as e:
            print(f"[Client {self.client_id}] Error getting server status: {e}")
            return None

    def make_prediction_request(self, features):
        """Make a prediction request to the server"""
        try:
            payload = {'features': features}
            response = requests.post(f"{self.server_url}/predict", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Client {self.client_id}] Prediction request failed.")
                return None
        except Exception as e:
            print(f"[Client {self.client_id}] Error making prediction request: {e}")
            return None


def run_client_simulation(num_clients=3, rounds=5):
    """Run a simulation with multiple clients"""
    print(f"Starting federated learning simulation with {num_clients} clients for {rounds} rounds...")
    
    clients = []
    for i in range(num_clients):
        client = ClientNode(f"client_{i+1}")
        clients.append(client)
    
    for round_num in range(rounds):
        print(f"\n=== Round {round_num + 1}/{rounds} ===")
        
        # All clients participate in this round
        for client in clients:
            success = client.run_federated_round(epochs=3, lr=0.001)
            if not success:
                print(f"Client {client.client_id} failed in round {round_num + 1}")
        
        # Wait a bit between rounds
        time.sleep(2)
    
    print("\n=== Simulation Complete ===")
    
    # Get final server status
    if clients:
        status = clients[0].get_server_status()
        print(f"Final server status: {status}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
        client = ClientNode(client_id)
        client.run_federated_round()
    else:
        # Run simulation
        run_client_simulation(num_clients=3, rounds=3)
