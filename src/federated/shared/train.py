import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(model, dataloader, val_dataloader=None, device='cpu', epochs=10, lr=0.001, patience=5):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for X, y in dataloader:
            X, y = X.to(device), y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y.cpu().numpy())
        
        avg_train_loss = train_loss / len(dataloader)
        train_acc = accuracy_score(train_targets, train_preds)
        training_history['train_loss'].append(avg_train_loss)
        training_history['train_acc'].append(train_acc)
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for X, y in val_dataloader:
                    X, y = X.to(device), y.float().unsqueeze(1).to(device)
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(y.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_dataloader)
            val_acc = accuracy_score(val_targets, val_preds)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_acc'].append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        model.train()
    
    return model.state_dict(), training_history

def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().unsqueeze(1).to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'targets': all_targets
    }
