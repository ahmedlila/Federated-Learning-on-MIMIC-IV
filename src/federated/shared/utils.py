import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def get_model_weights(model):
    return {k: v.cpu() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    model.load_state_dict(weights)

def load_data_for_client(client_id, data_dir="data/processed/data_preprocessing", test_size=0.2, random_state=42):
    """
    Load and split data for a specific client in federated learning
    """
    # Load data
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    X_val = pd.read_csv(f"{data_dir}/X_val.csv")
    y_val = pd.read_csv(f"{data_dir}/y_val.csv")
    
    # Combine train and validation data for client-specific splitting
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    
    # Use client_id to create reproducible splits
    np.random.seed(random_state + hash(client_id) % 1000)
    
    # Split data for this client
    X_client, X_test, y_client, y_test = train_test_split(
        X_combined, y_combined, test_size=test_size, random_state=random_state + hash(client_id) % 1000
    )
    
    # Further split client data into train and validation
    X_train_client, X_val_client, y_train_client, y_val_client = train_test_split(
        X_client, y_client, test_size=0.2, random_state=random_state + hash(client_id) % 1000
    )
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_client.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_client.values, dtype=torch.float32).squeeze()
    X_val_tensor = torch.tensor(X_val_client.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_client.values, dtype=torch.float32).squeeze()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze()
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'test': (X_test_tensor, y_test_tensor)
    }

def create_dataloaders(X_train, y_train, X_val=None, y_val=None, batch_size=32, shuffle=True):
    """
    Create DataLoaders for training and validation
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def federated_average(weights_list, client_sizes=None):
    """
    Perform federated averaging of model weights
    If client_sizes is provided, perform weighted averaging based on data size
    """
    if not weights_list:
        return None
    
    if client_sizes is None:
        # Simple averaging
        avg_weights = {}
        for key in weights_list[0].keys():
            avg_weights[key] = sum([w[key] for w in weights_list]) / len(weights_list)
    else:
        # Weighted averaging based on client data size
        total_size = sum(client_sizes)
        avg_weights = {}
        for key in weights_list[0].keys():
            weighted_sum = sum([w[key] * size for w, size in zip(weights_list, client_sizes)])
            avg_weights[key] = weighted_sum / total_size
    
    return avg_weights

def calculate_model_size(model):
    """Calculate the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def save_model_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
