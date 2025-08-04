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
    def __init__(self, client_id, server_url='http://localhost:5000', device='cpu'):
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
            response = requests.get(f"{self.server_url}/weights")
            if response.status_code == 200:
                weights = pickle.loads(response.content)
                set_model_weights(self.model, weights)
                print(f"[Client {self.client_id}] Global weights loaded successfully.")
                return True
            else:
                print(f"[Client {self.client_id}] Failed to fetch weights. Status: {response.status_code}")
                return False
        except Exception as e:
            print(f"[Client {self.client_id}] Error fetching weights: {e}")
            return False

    def load_local_data(self, data_dir="data/processed/data_preprocessing"):
        """Load and prepare local data for this client"""
        try:
            print(f"[Client {self.client_id}] Loading local data...")
            
            # Load data specific to this client
            client_data = load_data_for_client(self.client_id, data_dir)
            
            X_train, y_train = client_data['train']
            X_val, y_val = client_data['val']
            X_test, y_test = client_data['test']
            
            print(f"[Client {self.client_id}] Data loaded - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                X_train, y_train, X_val, y_val, batch_size=16, shuffle=True
            )
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'X_test': X_test,
                'y_test': y_test,
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
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
        
        # Step 3: Evaluate model before training
        test_dataset = TensorDataset(data_info['X_test'], data_info['y_test'])
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        pre_train_metrics = self.evaluate_local_model(test_loader)
        
        # Step 4: Local training
        updated_weights = self.local_train(
            data_info['train_loader'], 
            data_info['val_loader'], 
            epochs=epochs, 
            lr=lr
        )
        
        if updated_weights is None:
            print(f"[Client {self.client_id}] Training failed. Aborting round.")
            return False
        
        # Step 5: Evaluate model after training
        post_train_metrics = self.evaluate_local_model(test_loader)
        
        # Step 6: Send updated weights to server
        result = self.send_updated_weights(updated_weights, data_info['train_size'])
        
        if result:
            print(f"[Client {self.client_id}] Federated round completed successfully.")
            print(f"[Client {self.client_id}] Pre-training metrics: {pre_train_metrics}")
            print(f"[Client {self.client_id}] Post-training metrics: {post_train_metrics}")
            return True
        else:
            print(f"[Client {self.client_id}] Federated round failed.")
            return False

    def get_server_status(self):
        """Get current server status"""
        try:
            response = requests.get(f"{self.server_url}/status")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Client {self.client_id}] Failed to get server status.")
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
