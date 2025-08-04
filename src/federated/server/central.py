from flask import Flask, request, send_file, jsonify
import pickle, io
import torch
import json
import os
from datetime import datetime
from shared.model import AdmissionClassifier
from shared.utils import get_model_weights, set_model_weights, federated_average, evaluate_model
from shared.train import evaluate_model as evaluate_model_func
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

app = Flask(__name__)

# Global variables
model = AdmissionClassifier(input_dim=29)
clients_updates = []
client_sizes = []
round_number = 0
training_history = []
model_evaluation = {}

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

@app.route('/weights', methods=['GET'])
def send_weights():
    """Send current global model weights to clients"""
    buffer = pickle.dumps(get_model_weights(model))
    return send_file(io.BytesIO(buffer), mimetype='application/octet-stream')

@app.route('/update', methods=['POST'])
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

@app.route('/predict', methods=['POST'])
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

@app.route('/evaluate', methods=['POST'])
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

@app.route('/status', methods=['GET'])
def get_status():
    """Get current server status"""
    return jsonify({
        'round_number': round_number,
        'pending_updates': len(clients_updates),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'training_history': training_history,
        'model_evaluation': model_evaluation
    })

@app.route('/model_info', methods=['GET'])
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
        # Load test data
        X_test = pd.read_csv("data/processed/data_preprocessing/X_test.csv")
        y_test = pd.read_csv("data/processed/data_preprocessing/y_test.csv")
        
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
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[Server] Model evaluation - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        # Save evaluation to file
        with open(f'logs/evaluation_round_{round_number}.json', 'w') as f:
            json.dump(model_evaluation[f'round_{round_number}'], f, indent=2)
            
    except Exception as e:
        print(f"[Server] Error evaluating model: {e}")

@app.route('/reset', methods=['POST'])
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
    app.run(host='0.0.0.0', port=5000, debug=True)
