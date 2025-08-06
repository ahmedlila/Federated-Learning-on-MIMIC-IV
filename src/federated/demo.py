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
    def __init__(self, num_clients=3, rounds=5, server_url='http://localhost:8080', use_non_iid=False):
        self.num_clients = num_clients
        self.rounds = rounds
        self.server_url = server_url
        self.use_non_iid = use_non_iid
        self.clients = []
        self.results = {
            'rounds': [],
            'client_metrics': [],
            'global_metrics': [],
            'testing_results': {
                'client_testing': [],
                'server_evaluations': [],
                'final_predictions': {}
            },
            'data_distribution': {
                'type': 'non_iid' if use_non_iid else 'iid',
                'client_distributions': {}
            }
        }
        
        # Create results and logs directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        if self.use_non_iid:
            self.logger.info(f"Non-IID data distribution enabled")
            print("🔄 Non-IID data distribution enabled")
        else:
            self.logger.info(f"IID data distribution (default)")
            print("📊 IID data distribution (default)")
    
    def setup_logging(self):
        """Setup comprehensive logging for the federated learning demo"""
        import logging
        from datetime import datetime
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup main demo logger
        self.logger = logging.getLogger('federated_demo')
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(f'logs/demo_{timestamp}.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Federated Learning Demo started - {self.num_clients} clients, {self.rounds} rounds")
    
    def log_client_testing_results(self, client_id, round_num, pre_metrics, post_metrics, train_size, test_size):
        """Log detailed client testing results"""
        if pre_metrics and post_metrics:
            improvement = {
                'accuracy_improvement': post_metrics['accuracy'] - pre_metrics['accuracy'],
                'f1_improvement': post_metrics['f1_score'] - pre_metrics['f1_score'],
                'precision_improvement': post_metrics['precision'] - pre_metrics['precision'],
                'recall_improvement': post_metrics['recall'] - pre_metrics['recall']
            }
            
            self.logger.info(f"Client {client_id} Round {round_num} Testing Results:")
            self.logger.info(f"  Pre-training:  Acc={pre_metrics['accuracy']:.4f}, F1={pre_metrics['f1_score']:.4f}")
            self.logger.info(f"  Post-training: Acc={post_metrics['accuracy']:.4f}, F1={post_metrics['f1_score']:.4f}")
            self.logger.info(f"  Improvement:   Acc={improvement['accuracy_improvement']:+.4f}, F1={improvement['f1_improvement']:+.4f}")
            self.logger.info(f"  Data: Train={train_size}, Test={test_size}")
            
            # Store for CSV export
            self.results['testing_results']['client_testing'].append({
                'client_id': client_id,
                'round': round_num,
                'pre_accuracy': pre_metrics['accuracy'],
                'pre_f1_score': pre_metrics['f1_score'],
                'pre_precision': pre_metrics['precision'],
                'pre_recall': pre_metrics['recall'],
                'post_accuracy': post_metrics['accuracy'],
                'post_f1_score': post_metrics['f1_score'],
                'post_precision': post_metrics['precision'],
                'post_recall': post_metrics['recall'],
                'accuracy_improvement': improvement['accuracy_improvement'],
                'f1_improvement': improvement['f1_improvement'],
                'precision_improvement': improvement['precision_improvement'],
                'recall_improvement': improvement['recall_improvement'],
                'train_size': train_size,
                'test_size': test_size,
                'timestamp': datetime.now().isoformat()
            })
    
    def log_server_evaluation(self, round_num, metrics):
        """Log server evaluation results"""
        if metrics:
            self.logger.info(f"Server Round {round_num} Evaluation:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            
            # Store for CSV export
            self.results['testing_results']['server_evaluations'].append({
                'round': round_num,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'timestamp': datetime.now().isoformat()
            })
    
    def log_final_predictions(self, final_results):
        """Log final prediction results"""
        if final_results:
            metrics = final_results['final_metrics']
            self.logger.info("Final Model Predictions:")
            self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            self.logger.info(f"  Precision: {metrics['precision']:.4f}")
            self.logger.info(f"  Recall: {metrics['recall']:.4f}")
            self.logger.info(f"  Test Samples: {final_results['test_samples']}")
            
            # Log sample predictions
            self.logger.info("Sample Predictions:")
            for sample in final_results['sample_predictions'][:5]:
                status = "CORRECT" if sample['true_label'] == sample['predicted_label'] else "WRONG"
                self.logger.info(f"  {status}: Sample {sample['sample_id']} - True={sample['true_label']}, Pred={sample['predicted_label']}, Conf={sample['probability']:.3f}")
            
            # Store for CSV export
            self.results['testing_results']['final_predictions'] = {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'test_samples': final_results['test_samples'],
                'total_parameters': final_results['model_info']['total_parameters'],
                'rounds_completed': final_results['model_info']['rounds_completed'],
                'timestamp': final_results['model_info']['timestamp']
            }
    
    def export_results_to_csv(self):
        """Export all results to CSV files"""
        print("📊 Exporting results to CSV files...")
        self.logger.info("Exporting results to CSV files")
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # 1. Client Testing Results CSV
        if self.results['testing_results']['client_testing']:
            client_df = pd.DataFrame(self.results['testing_results']['client_testing'])
            client_csv_path = 'logs/client_testing_results.csv'
            client_df.to_csv(client_csv_path, index=False)
            print(f"✅ Client testing results saved to {client_csv_path}")
            self.logger.info(f"Client testing results saved to {client_csv_path}")
        
        # 2. Server Evaluation Results CSV
        if self.results['testing_results']['server_evaluations']:
            server_df = pd.DataFrame(self.results['testing_results']['server_evaluations'])
            server_csv_path = 'logs/server_evaluation_results.csv'
            server_df.to_csv(server_csv_path, index=False)
            print(f"✅ Server evaluation results saved to {server_csv_path}")
            self.logger.info(f"Server evaluation results saved to {server_csv_path}")
        
        # 3. Final Predictions CSV
        if self.results['testing_results']['final_predictions']:
            final_df = pd.DataFrame([self.results['testing_results']['final_predictions']])
            final_csv_path = 'logs/final_predictions_results.csv'
            final_df.to_csv(final_csv_path, index=False)
            print(f"✅ Final predictions results saved to {final_csv_path}")
            self.logger.info(f"Final predictions results saved to {final_csv_path}")
        
        # 4. Merged Evaluation Table (Client + Server)
        if self.results['testing_results']['client_testing'] or self.results['testing_results']['server_evaluations']:
            merged_data = []
            
            # Add client evaluations
            for client_result in self.results['testing_results']['client_testing']:
                merged_data.append({
                    'round': client_result['round'],
                    'client_id': client_result['client_id'],
                    'evaluation_type': 'client_post_training',
                    'accuracy': client_result['post_accuracy'],
                    'f1_score': client_result['post_f1_score'],
                    'precision': client_result['post_precision'],
                    'recall': client_result['post_recall'],
                    'train_size': client_result['train_size'],
                    'test_size': client_result['test_size'],
                    'timestamp': client_result['timestamp']
                })
            
            # Add server evaluations
            for server_result in self.results['testing_results']['server_evaluations']:
                merged_data.append({
                    'round': server_result['round'],
                    'client_id': 'server',
                    'evaluation_type': 'server_aggregation',
                    'accuracy': server_result['accuracy'],
                    'f1_score': server_result['f1_score'],
                    'precision': server_result['precision'],
                    'recall': server_result['recall'],
                    'train_size': None,
                    'test_size': None,
                    'timestamp': server_result['timestamp']
                })
            
            # Add final predictions
            if self.results['testing_results']['final_predictions']:
                final_result = self.results['testing_results']['final_predictions']
                merged_data.append({
                    'round': 'final',
                    'client_id': 'global_model',
                    'evaluation_type': 'final_evaluation',
                    'accuracy': final_result['accuracy'],
                    'f1_score': final_result['f1_score'],
                    'precision': final_result['precision'],
                    'recall': final_result['recall'],
                    'train_size': None,
                    'test_size': final_result['test_samples'],
                    'timestamp': final_result['timestamp']
                })
            
            # Create merged DataFrame and save
            merged_df = pd.DataFrame(merged_data)
            merged_csv_path = 'logs/merged_evaluation_results.csv'
            merged_df.to_csv(merged_csv_path, index=False)
            print(f"✅ Merged evaluation results saved to {merged_csv_path}")
            self.logger.info(f"Merged evaluation results saved to {merged_csv_path}")
        
        # 5. Non-IID Distribution CSV (if enabled)
        if self.use_non_iid and self.results['data_distribution']['client_distributions']:
            non_iid_data = []
            for client_id, dist_info in self.results['data_distribution']['client_distributions'].items():
                non_iid_data.append({
                    'client_id': client_id,
                    'distribution_type': dist_info['distribution_type'],
                    'train_samples': dist_info['train_samples'],
                    'val_samples': dist_info['val_samples'],
                    'test_samples': dist_info['test_samples'],
                    'train_positive_ratio': dist_info['train_label_distribution'].get(1, 0) / sum(dist_info['train_label_distribution'].values()),
                    'train_negative_ratio': dist_info['train_label_distribution'].get(0, 0) / sum(dist_info['train_label_distribution'].values()),
                    'data_path': dist_info['data_path']
                })
            
            non_iid_df = pd.DataFrame(non_iid_data)
            non_iid_csv_path = 'logs/non_iid_distribution_results.csv'
            non_iid_df.to_csv(non_iid_csv_path, index=False)
            print(f"✅ Non-IID distribution results saved to {non_iid_csv_path}")
            self.logger.info(f"Non-IID distribution results saved to {non_iid_csv_path}")
        
        # 6. Round Summary CSV
        if self.results['rounds']:
            round_data = []
            for round_result in self.results['rounds']:
                successful_clients = sum(1 for cr in round_result['client_results'] if cr['status'] == 'success')
                failed_clients = sum(1 for cr in round_result['client_results'] if cr['status'] == 'failed')
                
                round_data.append({
                    'round': round_result['round'],
                    'timestamp': round_result['timestamp'],
                    'total_clients': len(round_result['client_results']),
                    'successful_clients': successful_clients,
                    'failed_clients': failed_clients,
                    'success_rate': successful_clients / len(round_result['client_results']) if round_result['client_results'] else 0
                })
            
            round_df = pd.DataFrame(round_data)
            round_csv_path = 'logs/round_summary_results.csv'
            round_df.to_csv(round_csv_path, index=False)
            print(f"✅ Round summary results saved to {round_csv_path}")
            self.logger.info(f"Round summary results saved to {round_csv_path}")
        
        # 7. Model Quality Evaluation CSV
        if self.results.get('model_quality_evaluation'):
            quality_data = []
            
            # Basic metrics
            if 'basic_metrics' in self.results['model_quality_evaluation']:
                basic_metrics = self.results['model_quality_evaluation']['basic_metrics']
                quality_data.append({
                    'evaluation_type': 'basic_metrics',
                    'accuracy': basic_metrics['accuracy'],
                    'f1_score': basic_metrics['f1_score'],
                    'precision': basic_metrics['precision'],
                    'recall': basic_metrics['recall'],
                    'timestamp': self.results['model_quality_evaluation']['evaluation_timestamp']
                })
            
            # Convergence metrics
            if ('convergence_analysis' in self.results['model_quality_evaluation'] and 
                'convergence_metrics' in self.results['model_quality_evaluation']['convergence_analysis']):
                conv_metrics = self.results['model_quality_evaluation']['convergence_analysis']['convergence_metrics']
                
                # Check if all required metrics are available
                required_conv_metrics = ['final_accuracy', 'final_f1', 'accuracy_improvement', 'f1_improvement', 
                                       'accuracy_stability', 'f1_stability', 'convergence_round', 'monotonic_improvement']
                if all(metric in conv_metrics for metric in required_conv_metrics):
                    quality_data.append({
                        'evaluation_type': 'convergence_metrics',
                        'final_accuracy': conv_metrics['final_accuracy'],
                        'final_f1': conv_metrics['final_f1'],
                        'accuracy_improvement': conv_metrics['accuracy_improvement'],
                        'f1_improvement': conv_metrics['f1_improvement'],
                        'accuracy_stability': conv_metrics['accuracy_stability'],
                        'f1_stability': conv_metrics['f1_stability'],
                        'convergence_round': conv_metrics['convergence_round'],
                        'monotonic_improvement': conv_metrics['monotonic_improvement'],
                        'timestamp': self.results['model_quality_evaluation']['evaluation_timestamp']
                    })
            
            # Confidence metrics
            if 'confidence_analysis' in self.results['model_quality_evaluation']:
                conf_analysis = self.results['model_quality_evaluation']['confidence_analysis']
                quality_data.append({
                    'evaluation_type': 'confidence_metrics',
                    'mean_confidence': conf_analysis['mean_confidence'],
                    'confidence_std': conf_analysis['confidence_std'],
                    'correct_confidence_mean': conf_analysis['correct_confidence_mean'],
                    'incorrect_confidence_mean': conf_analysis['incorrect_confidence_mean'],
                    'confidence_calibration': conf_analysis['confidence_calibration'],
                    'high_confidence_accuracy': conf_analysis['high_confidence_accuracy'],
                    'timestamp': self.results['model_quality_evaluation']['evaluation_timestamp']
                })
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                quality_csv_path = 'logs/model_quality_evaluation.csv'
                quality_df.to_csv(quality_csv_path, index=False)
                print(f"✅ Model quality evaluation saved to {quality_csv_path}")
                self.logger.info(f"Model quality evaluation saved to {quality_csv_path}")
        
        # 8. Data Quality Evaluation CSV
        if self.results.get('data_quality_evaluation'):
            data_quality = self.results['data_quality_evaluation']['overall_quality']
            
            data_quality_data = {
                'total_samples': data_quality['total_samples'],
                'train_samples': data_quality['train_samples'],
                'val_samples': data_quality['val_samples'],
                'test_samples': data_quality['test_samples'],
                'features': data_quality['features'],
                'missing_values': data_quality['missing_values'],
                'duplicate_rows': data_quality['duplicate_rows'],
                'positive_class_ratio': data_quality['class_balance']['positive_ratio'],
                'negative_class_ratio': data_quality['class_balance']['negative_ratio'],
                'timestamp': datetime.now().isoformat()
            }
            
            data_quality_df = pd.DataFrame([data_quality_data])
            data_quality_csv_path = 'logs/data_quality_evaluation.csv'
            data_quality_df.to_csv(data_quality_csv_path, index=False)
            print(f"✅ Data quality evaluation saved to {data_quality_csv_path}")
            self.logger.info(f"Data quality evaluation saved to {data_quality_csv_path}")
        
        # 9. Feature Importance CSV
        if self.results.get('model_quality_evaluation') and 'feature_importance' in self.results['model_quality_evaluation']:
            feature_importance = self.results['model_quality_evaluation']['feature_importance']
            
            feature_data = []
            for feature, importance in feature_importance.items():
                feature_data.append({
                    'feature_name': feature,
                    'importance_score': importance,
                    'rank': len([f for f in feature_importance.values() if f > importance]) + 1
                })
            
            feature_df = pd.DataFrame(feature_data)
            feature_csv_path = 'logs/feature_importance.csv'
            feature_df.to_csv(feature_csv_path, index=False)
            print(f"✅ Feature importance saved to {feature_csv_path}")
            self.logger.info(f"Feature importance saved to {feature_csv_path}")
        
        print("📊 All CSV exports completed!")
        self.logger.info("All CSV exports completed")
    
    def start_server(self):
        """Start the federated learning server"""
        print("🚀 Starting Federated Learning Server...")
        
        # Start server in a separate process
        server_process = subprocess.Popen([
            sys.executable, 'server/central.py'
        ], cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Wait for server to start (longer wait for debug mode)
        print("⏳ Waiting for server to initialize...")
        time.sleep(8)
        
        # Check if server is running with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                response = requests.get(f"{self.server_url}/status", headers=headers, timeout=5)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    return server_process
                elif response.status_code == 403:
                    print("✅ Server is running (403 is expected with CORS)")
                    return server_process
                else:
                    print(f"⚠️ Server responded with status {response.status_code}, retrying...")
            except requests.exceptions.ConnectionError:
                print(f"⚠️ Server not ready yet (attempt {attempt + 1}/{max_retries}), waiting...")
                time.sleep(2)
            except Exception as e:
                print(f"⚠️ Error connecting to server: {e}")
                time.sleep(2)
        
        print("❌ Server failed to start after multiple attempts")
        return None
    
    def initialize_clients(self):
        """Initialize client nodes"""
        print(f"👥 Initializing {self.num_clients} clients...")
        
        # Create non-IID distributions if enabled
        client_distributions = None
        if self.use_non_iid:
            client_distributions = self.create_non_iid_distributions()
        
        for i in range(self.num_clients):
            client_id = f"client_{i+1}"
            
            # Get the data path for this client (only for non-IID)
            data_path = None
            if self.use_non_iid and client_distributions and client_id in client_distributions:
                data_path = client_distributions[client_id]['data_path']
                distribution_info = client_distributions[client_id]
                print(f"📊 {client_id} using {distribution_info['distribution_type']} distribution")
                print(f"   Train: {distribution_info['train_samples']}, Val: {distribution_info['val_samples']}, Test: {distribution_info['test_samples']}")
                print(f"   Label distribution: {distribution_info['train_label_distribution']}")
            else:
                print(f"📊 {client_id} using IID distribution (default)")
            
            # Create client
            client = ClientNode(client_id, self.server_url)
            
            # Store the data path in the client for later use (only for non-IID)
            if self.use_non_iid:
                client.non_iid_data_path = data_path
            
            self.clients.append(client)
            print(f"✅ Client {i+1} initialized")
    
    def run_federated_round(self, round_num):
        """Run a single federated learning round"""
        print(f"\n🔄 Starting Round {round_num + 1}/{self.rounds}")
        self.logger.info(f"Starting Round {round_num + 1}/{self.rounds}")
        
        round_results = {
            'round': round_num + 1,
            'timestamp': datetime.now().isoformat(),
            'client_results': []
        }
        
        # All clients participate in this round
        for client in self.clients:
            print(f"\n📊 Client {client.client_id} participating...")
            self.logger.info(f"Client {client.client_id} starting round {round_num + 1}")
            
            # Run federated round
            success = client.run_federated_round(epochs=3, lr=0.001)
            
            if success:
                print(f"✅ Client {client.client_id} completed round successfully")
                
                # Log client testing results if available
                if client.training_history:
                    latest_round = client.training_history[-1]
                    if 'pre_training_metrics' in latest_round and 'post_training_metrics' in latest_round:
                        self.log_client_testing_results(
                            client.client_id,
                            round_num + 1,
                            latest_round['pre_training_metrics'],
                            latest_round['post_training_metrics'],
                            latest_round['train_size'],
                            latest_round['test_size']
                        )
                
                round_results['client_results'].append({
                    'client_id': client.client_id,
                    'status': 'success'
                })
            else:
                print(f"❌ Client {client.client_id} failed in round")
                self.logger.warning(f"Client {client.client_id} failed in round {round_num + 1}")
                round_results['client_results'].append({
                    'client_id': client.client_id,
                    'status': 'failed'
                })
        
        # Wait for server to aggregate
        time.sleep(2)
        
        # Get server status and evaluation
        try:
            status_response = requests.get(f"{self.server_url}/status")
            if status_response.status_code == 200:
                status = status_response.json()
                round_results['server_status'] = status
                print(f"📈 Round {round_num + 1} completed. Global model updated.")
                
                # Log server evaluation if available
                if 'model_evaluation' in status and f'round_{round_num}' in status['model_evaluation']:
                    server_metrics = status['model_evaluation'][f'round_{round_num}']['metrics']
                    self.log_server_evaluation(round_num + 1, server_metrics)
                    
            else:
                print("⚠️ Could not get server status")
                self.logger.warning(f"Could not get server status for round {round_num + 1}")
        except Exception as e:
            print(f"⚠️ Error getting server status: {e}")
            self.logger.error(f"Error getting server status for round {round_num + 1}: {e}")
        
        self.results['rounds'].append(round_results)
        
        return round_results
    
    def evaluate_global_model(self):
        """Evaluate the global model on test data"""
        print("\n🔍 Evaluating global model...")
        
        try:
            # Load test data
            X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
            y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
            
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
    
    def make_final_predictions(self):
        """Make final predictions using the global model after all rounds"""
        print("\n🎯 Making final predictions with global model...")
        self.logger.info("Making final predictions with global model")
        
        try:
            # Call the final predictions endpoint
            final_predictions_response = requests.get(f"{self.server_url}/final_predictions")
            
            if final_predictions_response.status_code == 200:
                final_results = final_predictions_response.json()
                
                # Display final metrics
                metrics = final_results['final_metrics']
                print(f"🏆 Final Model Performance:")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1 Score: {metrics['f1_score']:.4f}")
                print(f"   Precision: {metrics['precision']:.4f}")
                print(f"   Recall: {metrics['recall']:.4f}")
                print(f"   Test Samples: {final_results['test_samples']}")
                
                # Display sample predictions
                print(f"\n📋 Sample Predictions:")
                for i, sample in enumerate(final_results['sample_predictions'][:5]):
                    status = "✅" if sample['true_label'] == sample['predicted_label'] else "❌"
                    print(f"   {status} Sample {sample['sample_id']}: True={sample['true_label']}, Predicted={sample['predicted_label']}, Confidence={sample['probability']:.3f}")
                
                # Log final predictions
                self.log_final_predictions(final_results)
                
                # Store final results
                self.results['final_predictions'] = final_results
                
                print(f"💾 Final predictions saved to logs/final_predictions.json")
                
            else:
                print(f"❌ Failed to get final predictions. Status: {final_predictions_response.status_code}")
                self.logger.error(f"Failed to get final predictions. Status: {final_predictions_response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Error making final predictions: {e}")
            self.logger.error(f"Error making final predictions: {e}")
    
    def visualize_results(self):
        """Create comprehensive visualizations of the federated learning results"""
        print("\n📊 Creating comprehensive visualizations...")
        self.logger.info("Creating comprehensive visualizations")
        
        # Create figure with subplots for main dashboard
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Federated Learning System Performance Dashboard', fontsize=18, fontweight='bold')
        
        # Plot 1: Round completion status (top left)
        successful_rounds = sum(1 for r in self.results['rounds'] if all(
            cr['status'] == 'success' for cr in r['client_results']
        ))
        failed_rounds = len(self.results['rounds']) - successful_rounds
        
        axes[0, 0].pie([successful_rounds, failed_rounds], 
                       labels=['Successful Rounds', 'Failed Rounds'], 
                       autopct='%1.1f%%',
                       colors=['lightgreen', 'lightcoral'],
                       startangle=90)
        axes[0, 0].set_title('Round Completion Status\n(All Clients Completed)', fontweight='bold', fontsize=12)
        
        # Plot 2: Client participation (top center)
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
        
        axes[0, 1].bar(x - width/2, success_counts, width, label='Successful', color='lightgreen', alpha=0.8)
        axes[0, 1].bar(x + width/2, failed_counts, width, label='Failed', color='lightcoral', alpha=0.8)
        axes[0, 1].set_xlabel('Client Nodes', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Rounds', fontweight='bold')
        axes[0, 1].set_title('Client Participation by Round', fontweight='bold', fontsize=12)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(clients)
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Server queue status (top right)
        if self.results['rounds'] and 'server_status' in self.results['rounds'][0]:
            rounds = [r['round'] for r in self.results['rounds']]
            pending_updates = [r['server_status']['pending_updates'] for r in self.results['rounds']]
            
            axes[0, 2].plot(rounds, pending_updates, marker='o', linewidth=2, markersize=8, color='blue', label='Pending Updates')
            axes[0, 2].set_xlabel('Federated Learning Round', fontweight='bold')
            axes[0, 2].set_ylabel('Number of Pending Updates', fontweight='bold')
            axes[0, 2].set_title('Server Queue Status', fontweight='bold', fontsize=12)
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
            
            # Add value annotations
            for i, (round_num, pending) in enumerate(zip(rounds, pending_updates)):
                axes[0, 2].annotate(f'{pending}', (round_num, pending), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=9)
        
        # Plot 4: Client Testing - Accuracy Improvement (middle left)
        if self.results['testing_results']['client_testing']:
            client_improvements = {}
            for test_result in self.results['testing_results']['client_testing']:
                client_id = test_result['client_id']
                if client_id not in client_improvements:
                    client_improvements[client_id] = []
                client_improvements[client_id].append(test_result['accuracy_improvement'])
            
            # Plot average improvement per client
            clients_test = list(client_improvements.keys())
            avg_improvements = [np.mean(client_improvements[c]) for c in clients_test]
            
            bars = axes[1, 0].bar(clients_test, avg_improvements, color='skyblue', alpha=0.7)
            axes[1, 0].set_xlabel('Client Nodes', fontweight='bold')
            axes[1, 0].set_ylabel('Average Accuracy Improvement', fontweight='bold')
            axes[1, 0].set_title('Client Local Training Effectiveness\n(Average Accuracy Improvement per Round)', fontweight='bold', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value annotations
            for bar, improvement in zip(bars, avg_improvements):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{improvement:+.3f}', ha='center', va='bottom' if improvement >= 0 else 'top', fontsize=9)
        
        # Plot 5: Server Evaluation - Global Model Performance (middle center)
        if self.results['testing_results']['server_evaluations']:
            server_rounds = [eval_result['round'] for eval_result in self.results['testing_results']['server_evaluations']]
            server_accuracies = [eval_result['accuracy'] for eval_result in self.results['testing_results']['server_evaluations']]
            server_f1_scores = [eval_result['f1_score'] for eval_result in self.results['testing_results']['server_evaluations']]
            
            axes[1, 1].plot(server_rounds, server_accuracies, marker='o', linewidth=2, markersize=8, 
                           color='green', label='Accuracy', alpha=0.8)
            axes[1, 1].plot(server_rounds, server_f1_scores, marker='s', linewidth=2, markersize=8, 
                           color='orange', label='F1 Score', alpha=0.8)
            axes[1, 1].set_xlabel('Federated Learning Round', fontweight='bold')
            axes[1, 1].set_ylabel('Performance Metric', fontweight='bold')
            axes[1, 1].set_title('Global Model Performance Evolution', fontweight='bold', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            axes[1, 1].set_ylim(0, 1)
            
            # Add value annotations
            for i, (round_num, acc, f1) in enumerate(zip(server_rounds, server_accuracies, server_f1_scores)):
                axes[1, 1].annotate(f'{acc:.3f}', (round_num, acc), textcoords="offset points", 
                                  xytext=(0,10), ha='center', fontsize=8)
                axes[1, 1].annotate(f'{f1:.3f}', (round_num, f1), textcoords="offset points", 
                                  xytext=(0,-15), ha='center', fontsize=8)
        
        # Plot 6: Final Predictions - Confusion Matrix (middle right)
        if self.results['testing_results']['final_predictions']:
            final_results = self.results['testing_results']['final_predictions']
            if 'confusion_matrix' in final_results:
                conf_matrix = np.array(final_results['confusion_matrix'])
                
                # Create confusion matrix heatmap
                im = axes[1, 2].imshow(conf_matrix, cmap='Blues', alpha=0.8)
                axes[1, 2].set_title('Final Model Confusion Matrix', fontweight='bold', fontsize=12)
                axes[1, 2].set_xlabel('Predicted Label', fontweight='bold')
                axes[1, 2].set_ylabel('True Label', fontweight='bold')
                axes[1, 2].set_xticks([0, 1])
                axes[1, 2].set_yticks([0, 1])
                axes[1, 2].set_xticklabels(['No Admission', 'Admission'])
                axes[1, 2].set_yticklabels(['No Admission', 'Admission'])
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = axes[1, 2].text(j, i, conf_matrix[i, j], ha="center", va="center", 
                                              color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black", fontweight='bold')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[1, 2], shrink=0.8)
            else:
                # If no confusion matrix, show final metrics instead
                final_metrics = final_results
                metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
                metric_values = [final_metrics['accuracy'], final_metrics['f1_score'],
                               final_metrics['precision'], final_metrics['recall']]
                
                bars = axes[1, 2].bar(metric_names, metric_values, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
                axes[1, 2].set_ylabel('Score', fontweight='bold')
                axes[1, 2].set_title('Final Model Performance', fontweight='bold', fontsize=12)
                axes[1, 2].grid(True, alpha=0.3, axis='y')
                axes[1, 2].set_ylim(0, 1)
                
                # Add value annotations
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 7: Client Testing - F1 Score Evolution (bottom left)
        if self.results['testing_results']['client_testing']:
            # Group by client and round
            client_f1_evolution = {}
            for test_result in self.results['testing_results']['client_testing']:
                client_id = test_result['client_id']
                round_num = test_result['round']
                post_f1 = test_result['post_f1_score']  # Fix: Access post_f1_score directly
                
                if client_id not in client_f1_evolution:
                    client_f1_evolution[client_id] = {}
                client_f1_evolution[client_id][round_num] = post_f1
            
            # Plot evolution for each client
            for client_id, f1_data in client_f1_evolution.items():
                rounds = sorted(f1_data.keys())
                f1_scores = [f1_data[r] for r in rounds]
                axes[2, 0].plot(rounds, f1_scores, marker='o', linewidth=2, markersize=6, 
                               label=f'{client_id}', alpha=0.8)
            
            axes[2, 0].set_xlabel('Federated Learning Round', fontweight='bold')
            axes[2, 0].set_ylabel('F1 Score', fontweight='bold')
            axes[2, 0].set_title('Client F1 Score Evolution', fontweight='bold', fontsize=12)
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
            axes[2, 0].set_ylim(0, 1)
        
        # Plot 8: Final Model Metrics Comparison (bottom center)
        if self.results['testing_results']['final_predictions']:
            final_metrics = self.results['testing_results']['final_predictions']
            metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            metrics_values = [final_metrics['accuracy'], final_metrics['f1_score'], 
                            final_metrics['precision'], final_metrics['recall']]
            
            bars = axes[2, 1].bar(metrics_names, metrics_values, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
            axes[2, 1].set_xlabel('Performance Metrics', fontweight='bold')
            axes[2, 1].set_ylabel('Score', fontweight='bold')
            axes[2, 1].set_title('Final Model Performance Metrics', fontweight='bold', fontsize=12)
            axes[2, 1].grid(True, alpha=0.3, axis='y')
            axes[2, 1].set_ylim(0, 1)
            
            # Add value annotations
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 9: Sample Predictions Analysis (bottom right)
        if self.results['testing_results']['final_predictions']:
            # Note: sample_predictions are not stored in the flattened structure
            # This plot will be skipped if no sample_predictions are available
            pass
        
        # Add overall statistics as text
        total_rounds = len(self.results['rounds'])
        total_clients = len(self.clients)
        success_rate = (successful_rounds / total_rounds * 100) if total_rounds > 0 else 0
        
        # Calculate average final performance
        final_performance = "N/A"
        if self.results['testing_results']['final_predictions']:
            final_acc = self.results['testing_results']['final_predictions']['accuracy']
            final_f1 = self.results['testing_results']['final_predictions']['f1_score']
            final_performance = f"Acc: {final_acc:.3f}, F1: {final_f1:.3f}"
        
        fig.text(0.02, 0.02, f'Summary: {total_rounds} rounds, {total_clients} clients, {success_rate:.1f}% success rate | Final Performance: {final_performance}', 
                fontsize=11, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('results/federated_learning_results.png', dpi=300, bbox_inches='tight')
        print("📈 Comprehensive visualizations saved to results/federated_learning_results.png")
        
        # Create additional detailed visualizations
        self.create_detailed_visualizations()
        
        # Create non-IID distribution visualizations (only if enabled)
        if self.use_non_iid:
            self.visualize_non_iid_distributions()
        
        # Create quality assessment visualizations
        self.create_quality_visualizations()
        
        # Save results to JSON
        with open('results/federated_learning_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("💾 Results saved to results/federated_learning_results.json")
    
    def create_detailed_visualizations(self):
        """Create additional detailed visualizations for specific aspects"""
        print("📊 Creating detailed visualizations...")
        
        # 1. Client Testing Detailed Analysis
        if self.results['testing_results']['client_testing']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Client Testing Detailed Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Pre vs Post Training Accuracy
            client_data = {}
            for test_result in self.results['testing_results']['client_testing']:
                client_id = test_result['client_id']
                if client_id not in client_data:
                    client_data[client_id] = {'pre_acc': [], 'post_acc': [], 'rounds': []}
                client_data[client_id]['pre_acc'].append(test_result['pre_accuracy'])
                client_data[client_id]['post_acc'].append(test_result['post_accuracy'])
                client_data[client_id]['rounds'].append(test_result['round'])
            
            for client_id, data in client_data.items():
                axes[0, 0].plot(data['rounds'], data['pre_acc'], marker='o', label=f'{client_id} Pre', alpha=0.7)
                axes[0, 0].plot(data['rounds'], data['post_acc'], marker='s', label=f'{client_id} Post', alpha=0.7)
            
            axes[0, 0].set_xlabel('Round', fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
            axes[0, 0].set_title('Pre vs Post Training Accuracy', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, 1)
            
            # Plot 2: Improvement Distribution
            improvements = [test_result['accuracy_improvement'] 
                          for test_result in self.results['testing_results']['client_testing']]
            axes[0, 1].hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('Accuracy Improvement', fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontweight='bold')
            axes[0, 1].set_title('Distribution of Accuracy Improvements', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', label=f'Mean: {np.mean(improvements):.3f}')
            axes[0, 1].legend()
            
            # Plot 3: Client Performance Comparison
            client_avg_performance = {}
            for test_result in self.results['testing_results']['client_testing']:
                client_id = test_result['client_id']
                if client_id not in client_avg_performance:
                    client_avg_performance[client_id] = {'pre': [], 'post': []}
                client_avg_performance[client_id]['pre'].append(test_result['pre_f1_score'])
                client_avg_performance[client_id]['post'].append(test_result['post_f1_score'])
            
            clients = list(client_avg_performance.keys())
            avg_pre = [np.mean(client_avg_performance[c]['pre']) for c in clients]
            avg_post = [np.mean(client_avg_performance[c]['post']) for c in clients]
            
            x = np.arange(len(clients))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, avg_pre, width, label='Average Pre-Training', color='lightcoral', alpha=0.8)
            axes[1, 0].bar(x + width/2, avg_post, width, label='Average Post-Training', color='lightgreen', alpha=0.8)
            axes[1, 0].set_xlabel('Client Nodes', fontweight='bold')
            axes[1, 0].set_ylabel('Average F1 Score', fontweight='bold')
            axes[1, 0].set_title('Average Client Performance Comparison', fontweight='bold')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(clients)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Training Effectiveness Heatmap
            effectiveness_data = []
            for client_id in clients:
                client_rounds = [r for r in self.results['testing_results']['client_testing'] if r['client_id'] == client_id]
                for round_data in client_rounds:
                    effectiveness_data.append([
                        round_data['round'],
                        clients.index(client_id),
                        round_data['accuracy_improvement']
                    ])
            
            if effectiveness_data:
                effectiveness_matrix = np.zeros((len(clients), max(r[0] for r in effectiveness_data)))
                for round_num, client_idx, improvement in effectiveness_data:
                    effectiveness_matrix[client_idx, round_num-1] = improvement
                
                im = axes[1, 1].imshow(effectiveness_matrix, cmap='RdYlGn', aspect='auto')
                axes[1, 1].set_xlabel('Round', fontweight='bold')
                axes[1, 1].set_ylabel('Client', fontweight='bold')
                axes[1, 1].set_title('Training Effectiveness Heatmap\n(Accuracy Improvement)', fontweight='bold')
                axes[1, 1].set_yticks(range(len(clients)))
                axes[1, 1].set_yticklabels(clients)
                plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
            
            plt.tight_layout()
            plt.savefig('results/client_testing_analysis.png', dpi=300, bbox_inches='tight')
            print("📈 Client testing analysis saved to results/client_testing_analysis.png")
        
        # 2. Server Performance Analysis
        if self.results['testing_results']['server_evaluations']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Server Performance Analysis', fontsize=16, fontweight='bold')
            
            server_rounds = [eval_result['round'] for eval_result in self.results['testing_results']['server_evaluations']]
            metrics = ['accuracy', 'f1_score', 'precision', 'recall']
            
            # Plot all metrics evolution
            for i, metric in enumerate(metrics):
                values = [eval_result[metric] for eval_result in self.results['testing_results']['server_evaluations']]
                axes[0, 0].plot(server_rounds, values, marker='o', linewidth=2, markersize=6, 
                               label=metric.replace('_', ' ').title(), alpha=0.8)
            
            axes[0, 0].set_xlabel('Round', fontweight='bold')
            axes[0, 0].set_ylabel('Score', fontweight='bold')
            axes[0, 0].set_title('Server Metrics Evolution', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, 1)
            
            # Plot performance improvement
            if len(server_rounds) > 1:
                initial_metrics = {metric: self.results['testing_results']['server_evaluations'][0][metric] 
                                 for metric in metrics}
                final_metrics = {metric: self.results['testing_results']['server_evaluations'][-1][metric] 
                               for metric in metrics}
                
                metric_names = [metric.replace('_', ' ').title() for metric in metrics]
                improvements = [final_metrics[metric] - initial_metrics[metric] for metric in metrics]
                
                bars = axes[0, 1].bar(metric_names, improvements, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
                axes[0, 1].set_ylabel('Improvement', fontweight='bold')
                axes[0, 1].set_title('Overall Performance Improvement', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                
                # Add value annotations
                for bar, improvement in zip(bars, improvements):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{improvement:+.3f}', ha='center', va='bottom' if improvement >= 0 else 'top', fontsize=9)
            
            # Plot correlation between rounds
            if len(server_rounds) > 2:
                accuracies = [eval_result['metrics']['accuracy'] for eval_result in self.results['testing_results']['server_evaluations']]
                f1_scores = [eval_result['metrics']['f1_score'] for eval_result in self.results['testing_results']['server_evaluations']]
                
                axes[1, 0].scatter(accuracies, f1_scores, c=server_rounds, cmap='viridis', s=100, alpha=0.7)
                axes[1, 0].set_xlabel('Accuracy', fontweight='bold')
                axes[1, 0].set_ylabel('F1 Score', fontweight='bold')
                axes[1, 0].set_title('Accuracy vs F1 Score Correlation', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add round labels
                for i, round_num in enumerate(server_rounds):
                    axes[1, 0].annotate(f'R{round_num}', (accuracies[i], f1_scores[i]), 
                                      xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot final performance summary
            if self.results['testing_results']['final_predictions']:
                final_metrics = self.results['testing_results']['final_predictions']
                metric_names = ['accuracy', 'f1_score', 'precision', 'recall']
                metric_values = [final_metrics['accuracy'], final_metrics['f1_score'], 
                               final_metrics['precision'], final_metrics['recall']]
                
                bars = axes[1, 1].bar(metric_names, metric_values, color=['green', 'orange', 'blue', 'red'], alpha=0.7)
                axes[1, 1].set_xlabel('Final Metrics', fontweight='bold')
                axes[1, 1].set_ylabel('Score', fontweight='bold')
                axes[1, 1].set_title('Final Model Performance Summary', fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                axes[1, 1].set_ylim(0, 1)
                
                # Add value annotations
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('results/server_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("📈 Server performance analysis saved to results/server_performance_analysis.png")
        
        self.logger.info("Detailed visualizations completed")
    
    def create_non_iid_distributions(self):
        """Create non-IID data distributions for each client (only when enabled)"""
        if not self.use_non_iid:
            return None
            
        print("🔄 Creating non-IID data distributions...")
        self.logger.info("Creating non-IID data distributions for clients")
        
        try:
            # Load the original data
            X_train = pd.read_csv("../../data/processed/data_preprocessing/X_train.csv")
            y_train = pd.read_csv("../../data/processed/data_preprocessing/y_train.csv")
            X_val = pd.read_csv("../../data/processed/data_preprocessing/X_val.csv")
            y_val = pd.read_csv("../../data/processed/data_preprocessing/y_val.csv")
            X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
            y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
            
            # Combine all data for distribution analysis
            all_data = pd.concat([X_train, X_val, X_test], ignore_index=True)
            all_labels = pd.concat([y_train, y_val, y_test], ignore_index=True)
            
            client_distributions = {}
            
            for i in range(self.num_clients):
                client_id = f"client_{i+1}"
                
                # Create different sampling strategies for each client
                if i == 0:
                    # Client 1: Bias towards positive class (admission)
                    positive_indices = all_labels[all_labels.iloc[:, 0] == 1].index
                    negative_indices = all_labels[all_labels.iloc[:, 0] == 0].index
                    
                    # Sample 70% positive, 30% negative
                    n_positive = int(len(positive_indices) * 0.7)
                    n_negative = int(len(negative_indices) * 0.3)
                    
                    selected_positive = np.random.choice(positive_indices, n_positive, replace=False)
                    selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
                    selected_indices = np.concatenate([selected_positive, selected_negative])
                    
                    distribution_type = "positive_bias"
                    
                elif i == 1:
                    # Client 2: Bias towards negative class (no admission)
                    positive_indices = all_labels[all_labels.iloc[:, 0] == 1].index
                    negative_indices = all_labels[all_labels.iloc[:, 0] == 0].index
                    
                    # Sample 30% positive, 70% negative
                    n_positive = int(len(positive_indices) * 0.3)
                    n_negative = int(len(negative_indices) * 0.7)
                    
                    selected_positive = np.random.choice(positive_indices, n_positive, replace=False)
                    selected_negative = np.random.choice(negative_indices, n_negative, replace=False)
                    selected_indices = np.concatenate([selected_positive, selected_negative])
                    
                    distribution_type = "negative_bias"
                    
                else:
                    # Client 3+: Feature-based bias
                    feature_cols = all_data.columns
                    
                    if 'age' in feature_cols or any('age' in col.lower() for col in feature_cols):
                        age_col = [col for col in feature_cols if 'age' in col.lower()][0]
                        # Bias towards older patients
                        age_threshold = all_data[age_col].quantile(0.7)
                        selected_indices = all_data[all_data[age_col] > age_threshold].index
                        distribution_type = "age_bias"
                    else:
                        # Random feature bias
                        random_feature = np.random.choice(feature_cols)
                        feature_threshold = all_data[random_feature].quantile(0.6)
                        selected_indices = all_data[all_data[random_feature] > feature_threshold].index
                        distribution_type = f"feature_bias_{random_feature}"
                
                # Ensure we have enough data
                if len(selected_indices) < 100:
                    remaining_indices = set(range(len(all_data))) - set(selected_indices)
                    additional_needed = 100 - len(selected_indices)
                    additional_indices = np.random.choice(list(remaining_indices), 
                                                        min(additional_needed, len(remaining_indices)), 
                                                        replace=False)
                    selected_indices = np.concatenate([selected_indices, additional_indices])
                
                # Shuffle the selected indices
                selected_indices_list = selected_indices.tolist()  # Convert to list for shuffling
                np.random.shuffle(selected_indices_list)
                selected_indices = np.array(selected_indices_list)  # Convert back to numpy array
                
                # Split into train/val/test for this client
                n_samples = len(selected_indices)
                train_size = int(n_samples * 0.7)
                val_size = int(n_samples * 0.15)
                
                train_indices = selected_indices[:train_size]
                val_indices = selected_indices[train_size:train_size + val_size]
                test_indices = selected_indices[train_size + val_size:]
                
                # Create client-specific datasets
                client_X_train = all_data.iloc[train_indices].reset_index(drop=True)
                client_y_train = all_labels.iloc[train_indices].reset_index(drop=True)
                client_X_val = all_data.iloc[val_indices].reset_index(drop=True)
                client_y_val = all_labels.iloc[val_indices].reset_index(drop=True)
                client_X_test = all_data.iloc[test_indices].reset_index(drop=True)
                client_y_test = all_labels.iloc[test_indices].reset_index(drop=True)
                
                # Save client-specific data
                client_data_dir = f"../../data/processed/data_preprocessing/client_{i+1}"
                os.makedirs(client_data_dir, exist_ok=True)
                
                client_X_train.to_csv(f"{client_data_dir}/X_train.csv", index=False)
                client_y_train.to_csv(f"{client_data_dir}/y_train.csv", index=False)
                client_X_val.to_csv(f"{client_data_dir}/X_val.csv", index=False)
                client_y_val.to_csv(f"{client_data_dir}/y_val.csv", index=False)
                client_X_test.to_csv(f"{client_data_dir}/X_test.csv", index=False)
                client_y_test.to_csv(f"{client_data_dir}/y_test.csv", index=False)
                
                # Calculate distribution statistics
                train_label_dist = client_y_train.iloc[:, 0].value_counts().to_dict()
                val_label_dist = client_y_val.iloc[:, 0].value_counts().to_dict()
                test_label_dist = client_y_test.iloc[:, 0].value_counts().to_dict()
                
                client_distributions[client_id] = {
                    'distribution_type': distribution_type,
                    'train_samples': len(client_X_train),
                    'val_samples': len(client_X_val),
                    'test_samples': len(client_X_test),
                    'train_label_distribution': train_label_dist,
                    'val_label_distribution': val_label_dist,
                    'test_label_distribution': test_label_dist,
                    'data_path': client_data_dir
                }
                
                print(f"✅ {client_id}: {distribution_type} - Train: {len(client_X_train)}, Val: {len(client_X_val)}, Test: {len(client_X_test)}")
                print(f"   Label distribution: {train_label_dist}")
                
                self.logger.info(f"{client_id} non-IID distribution created: {distribution_type}")
                self.logger.info(f"  Train: {len(client_X_train)} samples, Labels: {train_label_dist}")
            
            # Store distribution information
            self.results['data_distribution']['client_distributions'] = client_distributions
            
            print("🎯 Non-IID data distributions created successfully!")
            self.logger.info("Non-IID data distributions created successfully")
            
            return client_distributions
            
        except Exception as e:
            print(f"❌ Error creating non-IID distributions: {e}")
            self.logger.error(f"Error creating non-IID distributions: {e}")
            return None
    
    def visualize_non_iid_distributions(self):
        """Create visualizations for non-IID data distributions (only when enabled)"""
        if not self.use_non_iid or not self.results['data_distribution']['client_distributions']:
            return
        
        print("📊 Creating non-IID distribution visualizations...")
        self.logger.info("Creating non-IID distribution visualizations")
        
        client_distributions = self.results['data_distribution']['client_distributions']
        
        # Create simple visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Non-IID Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Label distribution per client
        clients = list(client_distributions.keys())
        positive_ratios = []
        negative_ratios = []
        
        for client_id in clients:
            train_dist = client_distributions[client_id]['train_label_distribution']
            total = sum(train_dist.values())
            positive_ratio = train_dist.get(1, 0) / total if total > 0 else 0
            negative_ratio = train_dist.get(0, 0) / total if total > 0 else 0
            
            positive_ratios.append(positive_ratio)
            negative_ratios.append(negative_ratio)
        
        x = np.arange(len(clients))
        width = 0.35
        
        axes[0].bar(x - width/2, positive_ratios, width, label='Positive Class (Admission)', color='red', alpha=0.7)
        axes[0].bar(x + width/2, negative_ratios, width, label='Negative Class (No Admission)', color='blue', alpha=0.7)
        
        axes[0].set_xlabel('Client Nodes', fontweight='bold')
        axes[0].set_ylabel('Class Distribution Ratio', fontweight='bold')
        axes[0].set_title('Label Distribution per Client', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(clients)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim(0, 1)
        
        # Add value annotations
        for i, (pos, neg) in enumerate(zip(positive_ratios, negative_ratios)):
            axes[0].text(i - width/2, pos/2, f'{pos:.2f}', ha='center', va='center', fontweight='bold', color='white')
            axes[0].text(i + width/2, neg/2, f'{neg:.2f}', ha='center', va='center', fontweight='bold', color='white')
        
        # Plot 2: Distribution types
        distribution_types = [client_distributions[c]['distribution_type'] for c in clients]
        type_counts = {}
        for dist_type in distribution_types:
            type_counts[dist_type] = type_counts.get(dist_type, 0) + 1
        
        if type_counts:
            types = list(type_counts.keys())
            counts = list(type_counts.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            wedges, texts, autotexts = axes[1].pie(counts, labels=types, autopct='%1.1f%%', 
                                                  colors=colors, startangle=90)
            axes[1].set_title('Distribution Types', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/non_iid_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("📈 Non-IID distribution analysis saved to results/non_iid_distribution_analysis.png")
        
        self.logger.info("Non-IID distribution visualizations completed")
    
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
            
            # Final evaluation and predictions
            print("\n🏁 Final evaluation and predictions...")
            self.evaluate_global_model()
            self.make_final_predictions()
            
            # Comprehensive model quality evaluation
            self.evaluate_model_quality()
            
            # Create visualizations
            self.visualize_results()
            
            # Export results to CSV
            self.export_results_to_csv()
            
            print("\n🎉 Federated Learning Demo completed successfully!")
            print("📁 Check the 'results' directory for outputs.")
            print("📁 Check the 'logs' directory for detailed evaluation results.")
            
        except KeyboardInterrupt:
            print("\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during demo: {e}")
        finally:
            # Clean up
            if server_process:
                server_process.terminate()
                print("🛑 Server stopped")
    
    def create_quality_visualizations(self):
        """Create comprehensive visualizations for model and data quality assessment"""
        if not self.results.get('model_quality_evaluation'):
            return
        
        print("📊 Creating quality assessment visualizations...")
        self.logger.info("Creating quality assessment visualizations")
        
        # Create comprehensive quality dashboard
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Comprehensive Model & Data Quality Assessment', fontsize=18, fontweight='bold')
        
        # Plot 1: Confusion Matrix
        if 'confusion_matrix' in self.results['model_quality_evaluation']:
            conf_matrix = np.array(self.results['model_quality_evaluation']['confusion_matrix'])
            
            im = axes[0, 0].imshow(conf_matrix, cmap='Blues', alpha=0.8)
            axes[0, 0].set_title('Confusion Matrix', fontweight='bold', fontsize=12)
            axes[0, 0].set_xlabel('Predicted Label', fontweight='bold')
            axes[0, 0].set_ylabel('True Label', fontweight='bold')
            axes[0, 0].set_xticks([0, 1])
            axes[0, 0].set_yticks([0, 1])
            axes[0, 0].set_xticklabels(['No Admission', 'Admission'])
            axes[0, 0].set_yticklabels(['No Admission', 'Admission'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = axes[0, 0].text(j, i, conf_matrix[i, j], ha="center", va="center", 
                                          color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black", fontweight='bold')
            
            plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        
        # Plot 2: Feature Importance
        if 'feature_importance' in self.results['model_quality_evaluation']:
            feature_importance = self.results['model_quality_evaluation']['feature_importance']
            top_features = list(feature_importance.items())[:10]  # Top 10 features
            
            features, importance = zip(*top_features)
            bars = axes[0, 1].barh(range(len(features)), importance, color='skyblue', alpha=0.7)
            axes[0, 1].set_yticks(range(len(features)))
            axes[0, 1].set_yticklabels(features)
            axes[0, 1].set_xlabel('Feature Importance (|Correlation|)', fontweight='bold')
            axes[0, 1].set_title('Top 10 Feature Importance', fontweight='bold', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Add value annotations
            for i, (bar, imp) in enumerate(zip(bars, importance)):
                axes[0, 1].text(imp + 0.01, bar.get_y() + bar.get_height()/2, f'{imp:.3f}', 
                               ha='left', va='center', fontsize=8)
        
        # Plot 3: Prediction Confidence Distribution
        if 'sample_predictions' in self.results['model_quality_evaluation']:
            sample_predictions = self.results['model_quality_evaluation']['sample_predictions']
            confidences = [pred['probability'] for pred in sample_predictions]
            correct_predictions = [pred['probability'] for pred in sample_predictions 
                                 if pred['true_label'] == pred['predicted_label']]
            incorrect_predictions = [pred['probability'] for pred in sample_predictions 
                                   if pred['true_label'] != pred['predicted_label']]
            
            axes[0, 2].hist(correct_predictions, bins=15, alpha=0.7, label='Correct Predictions', 
                           color='green', edgecolor='black', density=True)
            axes[0, 2].hist(incorrect_predictions, bins=15, alpha=0.7, label='Incorrect Predictions', 
                           color='red', edgecolor='black', density=True)
            axes[0, 2].set_xlabel('Prediction Confidence', fontweight='bold')
            axes[0, 2].set_ylabel('Density', fontweight='bold')
            axes[0, 2].set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=12)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Model Convergence Analysis
        if self.results['testing_results']['server_evaluations']:
            server_evals = self.results['testing_results']['server_evaluations']
            rounds = [eval_result['round'] for eval_result in server_evals]
            accuracies = [eval_result['accuracy'] for eval_result in server_evals]
            f1_scores = [eval_result['f1_score'] for eval_result in server_evals]
            
            axes[1, 0].plot(rounds, accuracies, marker='o', linewidth=2, markersize=6, 
                           color='blue', label='Accuracy', alpha=0.8)
            axes[1, 0].plot(rounds, f1_scores, marker='s', linewidth=2, markersize=6, 
                           color='orange', label='F1 Score', alpha=0.8)
            axes[1, 0].set_xlabel('Federated Learning Round', fontweight='bold')
            axes[1, 0].set_ylabel('Performance Metric', fontweight='bold')
            axes[1, 0].set_title('Model Convergence Analysis', fontweight='bold', fontsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1)
        
        # Plot 5: Data Quality Metrics
        if 'data_quality_evaluation' in self.results:
            data_quality = self.results['data_quality_evaluation']['overall_quality']
            
            quality_metrics = {
                'Total Samples': data_quality['total_samples'],
                'Features': data_quality['features'],
                'Missing Values': data_quality['missing_values'],
                'Duplicate Rows': data_quality['duplicate_rows']
            }
            
            metrics_names = list(quality_metrics.keys())
            metrics_values = list(quality_metrics.values())
            
            bars = axes[1, 1].bar(metrics_names, metrics_values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
            axes[1, 1].set_xlabel('Quality Metrics', fontweight='bold')
            axes[1, 1].set_ylabel('Count', fontweight='bold')
            axes[1, 1].set_title('Data Quality Overview', fontweight='bold', fontsize=12)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value annotations
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 6: Class Balance Analysis
        if 'data_quality_evaluation' in self.results:
            class_balance = data_quality['class_balance']
            classes = ['Negative (No Admission)', 'Positive (Admission)']
            ratios = [class_balance['negative_ratio'], class_balance['positive_ratio']]
            colors = ['lightblue', 'lightcoral']
            
            wedges, texts, autotexts = axes[1, 2].pie(ratios, labels=classes, autopct='%1.1f%%', 
                                                     colors=colors, startangle=90)
            axes[1, 2].set_title('Class Distribution', fontweight='bold', fontsize=12)
        
        # Plot 7: Confidence Calibration
        if 'confidence_analysis' in self.results['model_quality_evaluation']:
            confidence_analysis = self.results['model_quality_evaluation']['confidence_analysis']
            
            calibration_metrics = {
                'Mean Confidence': confidence_analysis['mean_confidence'],
                'Confidence Std': confidence_analysis['confidence_std'],
                'Correct Conf Mean': confidence_analysis['correct_confidence_mean'],
                'Incorrect Conf Mean': confidence_analysis['incorrect_confidence_mean'],
                'Calibration Score': confidence_analysis['confidence_calibration'],
                'High Conf Accuracy': confidence_analysis['high_confidence_accuracy']
            }
            
            metrics_names = list(calibration_metrics.keys())
            metrics_values = list(calibration_metrics.values())
            
            bars = axes[2, 0].bar(metrics_names, metrics_values, color='purple', alpha=0.7)
            axes[2, 0].set_xlabel('Confidence Metrics', fontweight='bold')
            axes[2, 0].set_ylabel('Score', fontweight='bold')
            axes[2, 0].set_title('Prediction Confidence Analysis', fontweight='bold', fontsize=12)
            axes[2, 0].grid(True, alpha=0.3, axis='y')
            axes[2, 0].set_ylim(0, 1)
            
            # Rotate x-axis labels for better readability
            axes[2, 0].tick_params(axis='x', rotation=45)
            
            # Add value annotations
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                axes[2, 0].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 8: Convergence Metrics
        if ('convergence_analysis' in self.results['model_quality_evaluation'] and 
            'convergence_metrics' in self.results['model_quality_evaluation']['convergence_analysis']):
            convergence_metrics = self.results['model_quality_evaluation']['convergence_analysis']['convergence_metrics']
            
            # Check if all required metrics are available
            required_metrics = ['final_accuracy', 'final_f1', 'accuracy_improvement', 'f1_improvement', 'accuracy_stability', 'f1_stability']
            if all(metric in convergence_metrics for metric in required_metrics):
                conv_metrics = {
                    'Final Accuracy': convergence_metrics['final_accuracy'],
                    'Final F1': convergence_metrics['final_f1'],
                    'Accuracy Improvement': convergence_metrics['accuracy_improvement'],
                    'F1 Improvement': convergence_metrics['f1_improvement'],
                    'Accuracy Stability': 1 - convergence_metrics['accuracy_stability'],  # Invert for better visualization
                    'F1 Stability': 1 - convergence_metrics['f1_stability']
                }
                
                metrics_names = list(conv_metrics.keys())
                metrics_values = list(conv_metrics.values())
                
                bars = axes[2, 1].bar(metrics_names, metrics_values, color='teal', alpha=0.7)
                axes[2, 1].set_xlabel('Convergence Metrics', fontweight='bold')
                axes[2, 1].set_ylabel('Score', fontweight='bold')
                axes[2, 1].set_title('Model Convergence Analysis', fontweight='bold', fontsize=12)
                axes[2, 1].grid(True, alpha=0.3, axis='y')
                
                # Rotate x-axis labels
                axes[2, 1].tick_params(axis='x', rotation=45)
                
                # Add value annotations
                for bar, value in zip(bars, metrics_values):
                    height = bar.get_height()
                    axes[2, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                # If metrics are missing, show a message
                axes[2, 1].text(0.5, 0.5, 'Convergence metrics\nnot available', 
                               ha='center', va='center', transform=axes[2, 1].transAxes,
                               fontsize=12, fontweight='bold')
                axes[2, 1].set_title('Model Convergence Analysis', fontweight='bold', fontsize=12)
        
        # Plot 9: Overall Quality Score
        overall_score = self.calculate_overall_quality_score()
        
        # Create a gauge chart
        angles = np.linspace(0, np.pi, 100)
        score_angle = overall_score * np.pi
        
        axes[2, 2].plot(np.cos(angles), np.sin(angles), 'k-', linewidth=2)
        axes[2, 2].fill_between(np.cos(angles[:int(score_angle*100/np.pi)]), 
                               np.sin(angles[:int(score_angle*100/np.pi)]), 
                               color='green' if overall_score > 0.7 else 'orange' if overall_score > 0.5 else 'red', 
                               alpha=0.6)
        axes[2, 2].text(0, 0, f'{overall_score:.2f}', ha='center', va='center', fontsize=20, fontweight='bold')
        axes[2, 2].set_xlim(-1.2, 1.2)
        axes[2, 2].set_ylim(-1.2, 1.2)
        axes[2, 2].set_aspect('equal')
        axes[2, 2].set_title('Overall Quality Score', fontweight='bold', fontsize=12)
        axes[2, 2].axis('off')
        
        # Add quality level text
        if overall_score > 0.8:
            quality_level = "Excellent"
            color = "green"
        elif overall_score > 0.6:
            quality_level = "Good"
            color = "orange"
        else:
            quality_level = "Needs Improvement"
            color = "red"
        
        axes[2, 2].text(0, -0.3, quality_level, ha='center', va='center', fontsize=12, 
                       fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig('results/quality_assessment_dashboard.png', dpi=300, bbox_inches='tight')
        print("📈 Quality assessment dashboard saved to results/quality_assessment_dashboard.png")
        
        self.logger.info("Quality assessment visualizations completed")
    
    def calculate_overall_quality_score(self):
        """Calculate overall quality score based on all assessments"""
        score = 0
        total_components = 0
        
        # Basic metrics score (30%)
        if (self.results.get('model_quality_evaluation') and 
            'basic_metrics' in self.results['model_quality_evaluation']):
            metrics = self.results['model_quality_evaluation']['basic_metrics']
            basic_score = (metrics['accuracy'] + metrics['f1_score']) / 2
            score += basic_score * 0.3
            total_components += 0.3
        
        # Convergence score (25%)
        if (self.results.get('model_quality_evaluation') and 
            'convergence_analysis' in self.results['model_quality_evaluation']):
            conv_metrics = self.results['model_quality_evaluation']['convergence_analysis']['convergence_metrics']
            if 'accuracy_improvement' in conv_metrics:
                convergence_score = min(1.0, (conv_metrics['accuracy_improvement'] + 0.5) / 1.0)
                score += convergence_score * 0.25
                total_components += 0.25
        
        # Confidence score (25%)
        if (self.results.get('model_quality_evaluation') and 
            'confidence_analysis' in self.results['model_quality_evaluation']):
            conf_analysis = self.results['model_quality_evaluation']['confidence_analysis']
            confidence_score = (conf_analysis['confidence_calibration'] + conf_analysis['high_confidence_accuracy']) / 2
            score += confidence_score * 0.25
            total_components += 0.25
        
        # Data quality score (20%)
        if self.results.get('data_quality_evaluation'):
            data_quality = self.results['data_quality_evaluation']['overall_quality']
            data_score = 1.0 if data_quality['missing_values'] == 0 and data_quality['duplicate_rows'] == 0 else 0.7
            score += data_score * 0.2
            total_components += 0.2
        
        # Normalize by total components
        return score / total_components if total_components > 0 else 0
    
    def evaluate_model_quality(self):
        """Comprehensive model quality evaluation with additional metrics"""
        print("\n🔍 Performing comprehensive model quality evaluation...")
        self.logger.info("Starting comprehensive model quality evaluation")
        
        try:
            # Load test data
            X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
            y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
            
            # Get final predictions from server
            final_predictions_response = requests.get(f"{self.server_url}/final_predictions")
            
            if final_predictions_response.status_code == 200:
                final_results = final_predictions_response.json()
                
                # Check if final_metrics exists in the response
                if 'final_metrics' in final_results:
                    # Store comprehensive evaluation results
                    self.results['model_quality_evaluation'] = {
                        'basic_metrics': final_results['final_metrics'],
                        'confusion_matrix': final_results.get('confusion_matrix', []),
                        'sample_predictions': final_results.get('sample_predictions', []),
                        'model_info': final_results.get('model_info', {}),
                        'evaluation_timestamp': datetime.now().isoformat()
                    }
                    
                    # Perform additional quality assessments
                    self.assess_data_quality()
                    self.assess_model_convergence()
                    self.assess_feature_importance()
                    self.assess_prediction_confidence()
                    
                    print("✅ Comprehensive model quality evaluation completed")
                    self.logger.info("Comprehensive model quality evaluation completed")
                else:
                    print("⚠️ Final metrics not available in server response")
                    self.logger.warning("Final metrics not available in server response")
                    
            else:
                print(f"❌ Failed to get final predictions for quality evaluation")
                self.logger.error("Failed to get final predictions for quality evaluation")
                
        except Exception as e:
            print(f"⚠️ Error in model quality evaluation: {e}")
            self.logger.error(f"Error in model quality evaluation: {e}")
    
    def assess_data_quality(self):
        """Assess data quality and distribution characteristics"""
        print("📊 Assessing data quality...")
        self.logger.info("Assessing data quality")
        
        try:
            # Load all data
            X_train = pd.read_csv("../../data/processed/data_preprocessing/X_train.csv")
            y_train = pd.read_csv("../../data/processed/data_preprocessing/y_train.csv")
            X_val = pd.read_csv("../../data/processed/data_preprocessing/X_val.csv")
            y_val = pd.read_csv("../../data/processed/data_preprocessing/y_val.csv")
            X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
            y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
            
            # Combine all data
            all_data = pd.concat([X_train, X_val, X_test], ignore_index=True)
            all_labels = pd.concat([y_train, y_val, y_test], ignore_index=True)
            
            # Data quality metrics
            data_quality = {
                'total_samples': len(all_data),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'features': len(all_data.columns),
                'missing_values': all_data.isnull().sum().sum(),
                'duplicate_rows': all_data.duplicated().sum(),
                'class_balance': {
                    'positive_ratio': (all_labels.iloc[:, 0] == 1).mean(),
                    'negative_ratio': (all_labels.iloc[:, 0] == 0).mean()
                },
                'feature_statistics': {
                    'mean': all_data.mean().to_dict(),
                    'std': all_data.std().to_dict(),
                    'min': all_data.min().to_dict(),
                    'max': all_data.max().to_dict(),
                    'skewness': all_data.skew().to_dict(),
                    'kurtosis': all_data.kurtosis().to_dict()
                }
            }
            
            # Store data quality results
            if 'data_quality_evaluation' not in self.results:
                self.results['data_quality_evaluation'] = {}
            self.results['data_quality_evaluation']['overall_quality'] = data_quality
            
            print(f"✅ Data quality assessment completed - {data_quality['total_samples']} samples, {data_quality['features']} features")
            self.logger.info(f"Data quality assessment completed - {data_quality['total_samples']} samples")
            
        except Exception as e:
            print(f"⚠️ Error in data quality assessment: {e}")
            self.logger.error(f"Error in data quality assessment: {e}")
    
    def assess_model_convergence(self):
        """Assess model convergence and training stability"""
        print("📈 Assessing model convergence...")
        self.logger.info("Assessing model convergence")
        
        try:
            convergence_analysis = {
                'rounds_completed': len(self.results['testing_results']['server_evaluations']),
                'convergence_metrics': {}
            }
            
            if self.results['testing_results']['server_evaluations']:
                server_evals = self.results['testing_results']['server_evaluations']
                
                # Calculate convergence metrics
                accuracies = [eval_result['accuracy'] for eval_result in server_evals]
                f1_scores = [eval_result['f1_score'] for eval_result in server_evals]
                
                # Convergence indicators
                convergence_analysis['convergence_metrics'] = {
                    'final_accuracy': accuracies[-1] if accuracies else 0,
                    'final_f1': f1_scores[-1] if f1_scores else 0,
                    'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
                    'f1_improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0,
                    'accuracy_stability': np.std(accuracies[-3:]) if len(accuracies) >= 3 else np.std(accuracies),
                    'f1_stability': np.std(f1_scores[-3:]) if len(f1_scores) >= 3 else np.std(f1_scores),
                    'convergence_round': self.find_convergence_round(accuracies),
                    'monotonic_improvement': self.check_monotonic_improvement(accuracies)
                }
            
            # Store convergence results
            if 'model_quality_evaluation' not in self.results:
                self.results['model_quality_evaluation'] = {}
            self.results['model_quality_evaluation']['convergence_analysis'] = convergence_analysis
            
            print("✅ Model convergence assessment completed")
            self.logger.info("Model convergence assessment completed")
            
        except Exception as e:
            print(f"⚠️ Error in model convergence assessment: {e}")
            self.logger.error(f"Error in model convergence assessment: {e}")
    
    def find_convergence_round(self, accuracies, threshold=0.001):
        """Find the round where model converges (accuracy improvement < threshold)"""
        if len(accuracies) < 2:
            return 1
        
        for i in range(1, len(accuracies)):
            if abs(accuracies[i] - accuracies[i-1]) < threshold:
                return i + 1
        return len(accuracies)
    
    def check_monotonic_improvement(self, accuracies):
        """Check if accuracy shows monotonic improvement"""
        if len(accuracies) < 2:
            return True
        
        improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
        return all(imp >= 0 for imp in improvements)
    
    def assess_feature_importance(self):
        """Assess feature importance using correlation analysis"""
        print("🔍 Assessing feature importance...")
        self.logger.info("Assessing feature importance")
        
        try:
            # Load data
            X_train = pd.read_csv("../../data/processed/data_preprocessing/X_train.csv")
            y_train = pd.read_csv("../../data/processed/data_preprocessing/y_train.csv")
            
            # Calculate feature correlations with target
            correlations = {}
            for feature in X_train.columns:
                correlation = X_train[feature].corr(y_train.iloc[:, 0])
                correlations[feature] = abs(correlation)
            
            # Sort features by importance
            feature_importance = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
            
            # Store feature importance results
            if 'model_quality_evaluation' not in self.results:
                self.results['model_quality_evaluation'] = {}
            self.results['model_quality_evaluation']['feature_importance'] = feature_importance
            
            print(f"✅ Feature importance assessment completed - Top feature: {list(feature_importance.keys())[0]}")
            self.logger.info("Feature importance assessment completed")
            
        except Exception as e:
            print(f"⚠️ Error in feature importance assessment: {e}")
            self.logger.error(f"Error in feature importance assessment: {e}")
    
    def assess_prediction_confidence(self):
        """Assess prediction confidence and reliability"""
        print("🎯 Assessing prediction confidence...")
        self.logger.info("Assessing prediction confidence")
        
        try:
            if 'model_quality_evaluation' in self.results and 'sample_predictions' in self.results['model_quality_evaluation']:
                sample_predictions = self.results['model_quality_evaluation']['sample_predictions']
                
                confidences = [pred['probability'] for pred in sample_predictions]
                correct_predictions = [pred for pred in sample_predictions if pred['true_label'] == pred['predicted_label']]
                incorrect_predictions = [pred for pred in sample_predictions if pred['true_label'] != pred['predicted_label']]
                
                confidence_analysis = {
                    'mean_confidence': np.mean(confidences),
                    'confidence_std': np.std(confidences),
                    'correct_confidence_mean': np.mean([p['probability'] for p in correct_predictions]) if correct_predictions else 0,
                    'incorrect_confidence_mean': np.mean([p['probability'] for p in incorrect_predictions]) if incorrect_predictions else 0,
                    'confidence_calibration': self.calculate_confidence_calibration(sample_predictions),
                    'high_confidence_accuracy': self.calculate_high_confidence_accuracy(sample_predictions, threshold=0.8)
                }
                
                # Store confidence analysis
                if 'model_quality_evaluation' not in self.results:
                    self.results['model_quality_evaluation'] = {}
                self.results['model_quality_evaluation']['confidence_analysis'] = confidence_analysis
                
                print("✅ Prediction confidence assessment completed")
                self.logger.info("Prediction confidence assessment completed")
            
        except Exception as e:
            print(f"⚠️ Error in prediction confidence assessment: {e}")
            self.logger.error(f"Error in prediction confidence assessment: {e}")
    
    def calculate_confidence_calibration(self, predictions, bins=10):
        """Calculate confidence calibration (how well confidence matches accuracy)"""
        if not predictions:
            return 0
        
        # Group predictions by confidence bins
        confidences = [p['probability'] for p in predictions]
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        
        calibration_error = 0
        for i in range(bins):
            bin_predictions = [p for j, p in enumerate(predictions) if bin_indices[j] == i]
            if bin_predictions:
                avg_confidence = np.mean([p['probability'] for p in bin_predictions])
                accuracy = np.mean([1 if p['true_label'] == p['predicted_label'] else 0 for p in bin_predictions])
                calibration_error += abs(avg_confidence - accuracy)
        
        return 1 - (calibration_error / bins)  # Higher is better
    
    def calculate_high_confidence_accuracy(self, predictions, threshold=0.8):
        """Calculate accuracy for high-confidence predictions"""
        high_conf_predictions = [p for p in predictions if p['probability'] >= threshold]
        if not high_conf_predictions:
            return 0
        
        return np.mean([1 if p['true_label'] == p['predicted_label'] else 0 for p in high_conf_predictions])

def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Demo')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--rounds', type=int, default=5, help='Number of federated rounds')
    parser.add_argument('--server-url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--non-iid', default=False, action='store_true', help='Enable non-IID data distribution')
    
    args = parser.parse_args()
    
    # Run demo
    demo = FederatedLearningDemo(
        num_clients=args.clients,
        rounds=args.rounds,
        server_url=args.server_url,
        use_non_iid=args.non_iid
    )
    demo.run_demo()

if __name__ == '__main__':
    main() 