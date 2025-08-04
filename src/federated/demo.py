#!/usr/bin/env python3
"""
Federated Learning Demo for Hospital Admission Prediction
=======================================================

This script demonstrates a complete federated learning system where multiple clients
train a model on their local data and contribute to a global model without sharing
raw data.

Features:
- Multiple clients with different data splits
- Central server for model aggregation
- Real-time model evaluation
- Prediction capabilities
- Comprehensive logging and visualization
"""

import os
import sys
import time
import json
import threading
import subprocess
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add the federated directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client.node import ClientNode
from shared.utils import load_data_for_client

class FederatedLearningDemo:
    def __init__(self, num_clients=3, rounds=5, server_url='http://localhost:5000'):
        self.num_clients = num_clients
        self.rounds = rounds
        self.server_url = server_url
        self.clients = []
        self.results = {
            'rounds': [],
            'client_metrics': [],
            'global_metrics': []
        }
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
    def start_server(self):
        """Start the federated learning server"""
        print("🚀 Starting Federated Learning Server...")
        
        # Start server in a separate process
        server_process = subprocess.Popen([
            sys.executable, 'server/central.py'
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Wait for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get(f"{self.server_url}/status")
            if response.status_code == 200:
                print("✅ Server started successfully!")
                return server_process
            else:
                print("❌ Server failed to start")
                return None
        except Exception as e:
            print(f"❌ Error connecting to server: {e}")
            return None
    
    def initialize_clients(self):
        """Initialize client nodes"""
        print(f"👥 Initializing {self.num_clients} clients...")
        
        for i in range(self.num_clients):
            client = ClientNode(f"client_{i+1}", self.server_url)
            self.clients.append(client)
            print(f"✅ Client {i+1} initialized")
    
    def run_federated_round(self, round_num):
        """Run a single federated learning round"""
        print(f"\n🔄 Starting Round {round_num + 1}/{self.rounds}")
        
        round_results = {
            'round': round_num + 1,
            'timestamp': datetime.now().isoformat(),
            'client_results': []
        }
        
        # All clients participate in this round
        for client in self.clients:
            print(f"\n📊 Client {client.client_id} participating...")
            
            # Run federated round
            success = client.run_federated_round(epochs=3, lr=0.001)
            
            if success:
                print(f"✅ Client {client.client_id} completed round successfully")
                round_results['client_results'].append({
                    'client_id': client.client_id,
                    'status': 'success'
                })
            else:
                print(f"❌ Client {client.client_id} failed in round")
                round_results['client_results'].append({
                    'client_id': client.client_id,
                    'status': 'failed'
                })
        
        # Wait for server to aggregate
        time.sleep(2)
        
        # Get server status
        try:
            status_response = requests.get(f"{self.server_url}/status")
            if status_response.status_code == 200:
                status = status_response.json()
                round_results['server_status'] = status
                print(f"📈 Round {round_num + 1} completed. Global model updated.")
            else:
                print("⚠️ Could not get server status")
        except Exception as e:
            print(f"⚠️ Error getting server status: {e}")
        
        self.results['rounds'].append(round_results)
        
        return round_results
    
    def evaluate_global_model(self):
        """Evaluate the global model on test data"""
        print("\n🔍 Evaluating global model...")
        
        try:
            # Load test data
            X_test = pd.read_csv("data/processed/data_preprocessing/X_test.csv")
            y_test = pd.read_csv("data/processed/data_preprocessing/y_test.csv")
            
            # Make prediction request to server
            sample_features = X_test.iloc[0].values.tolist()
            prediction_response = requests.post(
                f"{self.server_url}/predict", 
                json={'features': sample_features}
            )
            
            if prediction_response.status_code == 200:
                prediction = prediction_response.json()
                print(f"🎯 Sample prediction: {prediction}")
            
            # Get model info
            model_info_response = requests.get(f"{self.server_url}/model_info")
            if model_info_response.status_code == 200:
                model_info = model_info_response.json()
                print(f"📊 Model info: {model_info}")
                
        except Exception as e:
            print(f"⚠️ Error evaluating global model: {e}")
    
    def visualize_results(self):
        """Create visualizations of the federated learning results"""
        print("\n📊 Creating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Federated Learning Results', fontsize=16)
        
        # Plot 1: Round completion status
        successful_rounds = sum(1 for r in self.results['rounds'] if all(
            cr['status'] == 'success' for cr in r['client_results']
        ))
        failed_rounds = len(self.results['rounds']) - successful_rounds
        
        axes[0, 0].pie([successful_rounds, failed_rounds], 
                       labels=['Successful', 'Failed'], 
                       autopct='%1.1f%%',
                       colors=['lightgreen', 'lightcoral'])
        axes[0, 0].set_title('Round Completion Status')
        
        # Plot 2: Client participation
        client_participation = {}
        for round_result in self.results['rounds']:
            for client_result in round_result['client_results']:
                client_id = client_result['client_id']
                if client_id not in client_participation:
                    client_participation[client_id] = {'success': 0, 'failed': 0}
                
                if client_result['status'] == 'success':
                    client_participation[client_id]['success'] += 1
                else:
                    client_participation[client_id]['failed'] += 1
        
        clients = list(client_participation.keys())
        success_counts = [client_participation[c]['success'] for c in clients]
        failed_counts = [client_participation[c]['failed'] for c in clients]
        
        x = np.arange(len(clients))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, success_counts, width, label='Successful', color='lightgreen')
        axes[0, 1].bar(x + width/2, failed_counts, width, label='Failed', color='lightcoral')
        axes[0, 1].set_xlabel('Clients')
        axes[0, 1].set_ylabel('Rounds')
        axes[0, 1].set_title('Client Participation by Round')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(clients)
        axes[0, 1].legend()
        
        # Plot 3: Server status over rounds
        if self.results['rounds'] and 'server_status' in self.results['rounds'][0]:
            rounds = [r['round'] for r in self.results['rounds']]
            pending_updates = [r['server_status']['pending_updates'] for r in self.results['rounds']]
            
            axes[1, 0].plot(rounds, pending_updates, marker='o', linewidth=2, markersize=8)
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Pending Updates')
            axes[1, 0].set_title('Server Status: Pending Updates')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Model parameters
        if self.results['rounds'] and 'server_status' in self.results['rounds'][0]:
            model_params = [r['server_status']['model_parameters'] for r in self.results['rounds']]
            
            axes[1, 1].bar(rounds, model_params, color='skyblue', alpha=0.7)
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Model Parameters')
            axes[1, 1].set_title('Model Complexity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/federated_learning_results.png', dpi=300, bbox_inches='tight')
        print("📈 Visualizations saved to results/federated_learning_results.png")
        
        # Save results to JSON
        with open('results/federated_learning_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("💾 Results saved to results/federated_learning_results.json")
    
    def run_demo(self):
        """Run the complete federated learning demo"""
        print("🎯 Starting Federated Learning Demo")
        print("=" * 50)
        
        # Start server
        server_process = self.start_server()
        if server_process is None:
            print("❌ Failed to start server. Exiting.")
            return
        
        try:
            # Initialize clients
            self.initialize_clients()
            
            # Run federated learning rounds
            for round_num in range(self.rounds):
                self.run_federated_round(round_num)
                
                # Evaluate global model after each round
                self.evaluate_global_model()
                
                # Wait between rounds
                if round_num < self.rounds - 1:
                    print("\n⏳ Waiting between rounds...")
                    time.sleep(3)
            
            # Final evaluation
            print("\n🏁 Final evaluation...")
            self.evaluate_global_model()
            
            # Create visualizations
            self.visualize_results()
            
            print("\n🎉 Federated Learning Demo completed successfully!")
            print("📁 Check the 'results' directory for outputs.")
            
        except KeyboardInterrupt:
            print("\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during demo: {e}")
        finally:
            # Clean up
            if server_process:
                server_process.terminate()
                print("🛑 Server stopped")

def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--server-url', default='http://localhost:5000', help='Server URL')
    
    args = parser.parse_args()
    
    # Run demo
    demo = FederatedLearningDemo(
        num_clients=args.clients,
        rounds=args.rounds,
        server_url=args.server_url
    )
    demo.run_demo()

if __name__ == '__main__':
    main() 