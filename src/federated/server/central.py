import sys
import os
# Add the federated directory to the path so we can import shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import pickle, io
import torch
import json
from datetime import datetime
from shared.model import AdmissionClassifier
from shared.utils import get_model_weights, set_model_weights, federated_average
from shared.train import evaluate_model as evaluate_model_func
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)
CORS(app, 
     resources={r"/*": {"origins": "*"}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,PATCH,OPTIONS,HEAD')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Global variables
model = AdmissionClassifier(input_dim=29)
clients_updates = []
client_sizes = []
round_number = 0
training_history = []
model_evaluation = {}

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

@app.route('/weights', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def send_weights():
    """Send current global model weights to clients"""
    print(f"[Server] Received weights request from {request.remote_addr}")
    try:
        buffer = pickle.dumps(get_model_weights(model))
        print(f"[Server] Sending weights successfully")
        return send_file(io.BytesIO(buffer), mimetype='application/octet-stream')
    except Exception as e:
        print(f"[Server] Error sending weights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/update', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def receive_update():
    """Receive model updates from clients"""
    global clients_updates, client_sizes, round_number, model
    
    data = request.get_json()
    weights = pickle.loads(data['weights'].encode('latin1'))
    client_size = data.get('client_size', 1)
    
    clients_updates.append(weights)
    client_sizes.append(client_size)
    
    print(f"[Server] Received update from client. Total updates: {len(clients_updates)}")
    
    # Check if we have enough updates to aggregate (configurable)
    min_clients = int(request.args.get('min_clients', 3))
    
    if len(clients_updates) >= min_clients:
        print(f"[Server] Aggregating {len(clients_updates)} client updates...")
        
        # Perform federated averaging
        avg_weights = federated_average(clients_updates, client_sizes)
        set_model_weights(model, avg_weights)
        
        # Evaluate global model
        evaluate_global_model()
        
        # Log the round
        round_info = {
            'round': round_number,
            'num_clients': len(clients_updates),
            'client_sizes': client_sizes.copy(),
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(round_info)
        
        # Reset for next round
        clients_updates = []
        client_sizes = []
        round_number += 1
        
        print(f"[Server] Round {round_number-1} completed. Global model updated.")
        
        return jsonify({
            'status': 'success',
            'message': f'Round {round_number-1} completed',
            'num_clients': len(clients_updates),
            'round_number': round_number
        })
    
    return jsonify({
        'status': 'waiting',
        'message': f'Waiting for more clients. Current: {len(clients_updates)}/{min_clients}',
        'num_clients': len(clients_updates)
    })

@app.route('/predict', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def predict():
    """Make predictions using the global model"""
    try:
        data = request.get_json()
        features = data['features']
        
        # Convert to tensor
        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).int()
        
        return jsonify({
            'prediction': int(predictions.item()),
            'probability': float(probabilities.item()),
            'features_used': len(features)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/evaluate', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def evaluate():
    """Evaluate the global model on test data"""
    try:
        data = request.get_json()
        X_test = torch.tensor(data['X_test'], dtype=torch.float32)
        y_test = torch.tensor(data['y_test'], dtype=torch.float32)
        
        # Create dataloader
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate model
        metrics = evaluate_model_func(model, test_loader)
        
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/status', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def get_status():
    """Get current server status"""
    print(f"[Server] Received status request from {request.remote_addr}")
    try:
        # Convert numpy arrays and other non-serializable objects
        serializable_training_history = []
        for round_info in training_history:
            serializable_round = {
                'round': round_info['round'],
                'num_clients': round_info['num_clients'],
                'client_sizes': [int(size) for size in round_info['client_sizes']],  # Convert numpy to int
                'timestamp': round_info['timestamp']
            }
            serializable_training_history.append(serializable_round)
        
        serializable_model_evaluation = {}
        if model_evaluation:
            for key, value in model_evaluation.items():
                if isinstance(value, (np.ndarray, np.integer, np.floating)):
                    serializable_model_evaluation[key] = value.item() if value.size == 1 else value.tolist()
                else:
                    serializable_model_evaluation[key] = value
        
        status = {
            'round_number': round_number,
            'pending_updates': len(clients_updates),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'training_history': serializable_training_history,
            'model_evaluation': serializable_model_evaluation
        }
        print(f"[Server] Sending status successfully")
        return jsonify(status)
    except Exception as e:
        print(f"[Server] Error sending status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def get_model_info():
    """Get information about the current model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'model_type': 'AdmissionClassifier',
        'input_dim': 29,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_structure': str(model)
    })

def evaluate_global_model():
    """Evaluate the global model on test data"""
    global model_evaluation
    
    try:
        print(f"[Server] Evaluating global model after round {round_number}...")
        
        # Load test data
        X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
        y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
        
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()
        
        # Create dataloader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        metrics = evaluate_model_func(model, test_loader)
        
        # Store evaluation results
        model_evaluation[f'round_{round_number}'] = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        
        print(f"[Server] Global model evaluation - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        # Save evaluation to CSV
        evaluation_data = {
            'round': round_number,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'test_samples': len(X_test),
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Create or append to CSV file
        csv_path = f'logs/server_evaluations.csv'
        os.makedirs('logs', exist_ok=True)
        
        if os.path.exists(csv_path):
            # Append to existing file using pd.concat (df.append is deprecated)
            df = pd.read_csv(csv_path)
            new_df = pd.DataFrame([evaluation_data])
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([evaluation_data])
        
        df.to_csv(csv_path, index=False)
        print(f"[Server] Evaluation results saved to {csv_path}")
            
        return metrics
            
    except Exception as e:
        print(f"[Server] Error evaluating model: {e}")
        return None

@app.route('/final_predictions', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def final_predictions():
    """Make final predictions using the global model after all rounds"""
    try:
        print(f"[Server] Making final predictions with global model...")
        
        # Load test data
        X_test = pd.read_csv("../../data/processed/data_preprocessing/X_test.csv")
        y_test = pd.read_csv("../../data/processed/data_preprocessing/y_test.csv")
        
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()
        
        # Create dataloader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).int()
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_probabilities.extend(probabilities.cpu().numpy().flatten())
                all_true_labels.extend(batch_y.cpu().numpy().flatten())
        
        # Calculate final metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions)
        precision = precision_score(all_true_labels, all_predictions)
        recall = recall_score(all_true_labels, all_predictions)
        conf_matrix = confusion_matrix(all_true_labels, all_predictions).tolist()
        
        # Create sample predictions for demonstration
        sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        sample_predictions = []
        
        for idx in sample_indices:
            sample_predictions.append({
                'sample_id': int(idx),
                'true_label': int(all_true_labels[idx]),
                'predicted_label': int(all_predictions[idx]),
                'probability': float(all_probabilities[idx]),
                'features': X_test.iloc[idx].values.tolist()
            })
        
        final_results = {
            'final_metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            },
            'confusion_matrix': conf_matrix,
            'test_samples': len(X_test),
            'sample_predictions': sample_predictions,
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'rounds_completed': round_number,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        print(f"[Server] Final predictions completed - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save final results to CSV
        final_results_csv = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'test_samples': len(X_test),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'rounds_completed': round_number,
            'timestamp': datetime.now().isoformat()
        }
        
        csv_path = 'logs/final_predictions.csv'
        os.makedirs('logs', exist_ok=True)
        
        # Save final results
        final_df = pd.DataFrame([final_results_csv])
        final_df.to_csv(csv_path, index=False)
        print(f"[Server] Final predictions saved to {csv_path}")
        
        return jsonify(final_results)
        
    except Exception as e:
        print(f"[Server] Error making final predictions: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/reset', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD'])
def reset_model():
    """Reset the global model to initial state"""
    global model, clients_updates, client_sizes, round_number, training_history, model_evaluation
    
    model = AdmissionClassifier(input_dim=29)
    clients_updates = []
    client_sizes = []
    round_number = 0
    training_history = []
    model_evaluation = {}
    
    return jsonify({'status': 'success', 'message': 'Model reset successfully'})

if __name__ == '__main__':
    print("[Server] Starting Federated Learning Server...")
    print(f"[Server] Model parameters: {sum(p.numel() for p in model.parameters())}")
    app.run(host='0.0.0.0', port=8080, debug=False)
