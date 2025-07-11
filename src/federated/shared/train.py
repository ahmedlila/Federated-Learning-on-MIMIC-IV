import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataloader, device='cpu', epochs=5, lr=0.001):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for X, y in dataloader:
            X, y = X.to(device), y.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()
