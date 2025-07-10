import torch
import pickle
import requests
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from shared.model import AdmissionClassifier
from shared.train import train
from shared.utils import set_model_weights, get_model_weights

class ClientNode:
    def __init__(self, client_id, input_dim=29, device='cpu'):
        self.client_id = client_id
        self.input_dim = input_dim
        self.device = device
        self.model = AdmissionClassifier(input_dim=input_dim).to(self.device)

    def fetch_global_weights(self, url='http://central:5000/weights'):
        print("[Client] Fetching global model weights...")
        response = requests.get(url)
        weights = pickle.loads(response.content)
        set_model_weights(self.model, weights)
        print("[Client] Weights loaded.")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            logits = self.model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
        return preds.cpu().numpy()

    def local_train(self, X, y, epochs=5, lr=0.001):
        loader = DataLoader(TensorDataset(X, y), batch_size=1)
        updated_weights = train(self.model, loader, device=self.device, epochs=epochs, lr=lr)
        return updated_weights

    def send_updated_weights(self, weights, url='http://central:5000/update'):
        print("[Client] Sending updated weights to central server...")
        payload = pickle.dumps(weights)
        response = requests.post(url, data=payload)
        print(f"[Client] Update status: {response.status_code}")
        return response.status_code

    def run_round(self, data_path_x="X_test.csv", data_path_y="y_test.csv"):
        print(f"[Client {self.client_id}] Starting federated round...")

        # Step 1: Load real data (simulate single row training)
        X_df = pd.read_csv(data_path_x)
        y_df = pd.read_csv(data_path_y)

        X_tensor = torch.tensor(X_df.iloc[[0]].values, dtype=torch.float32)
        y_tensor = torch.tensor(y_df.iloc[[0]].values, dtype=torch.float32).squeeze()

        # Step 2: Fetch global model
        self.fetch_global_weights()

        # Step 3: Predict before training
        preds = self.predict(X_tensor)
        print(f"[Client {self.client_id}] Prediction before training: {preds[0]}")

        # Step 4: Local training
        updated_weights = self.local_train(X_tensor, y_tensor)

        # Step 5: Send back to central
        self.send_updated_weights(updated_weights)


# 🔧 Entry Point
if __name__ == '__main__':
    client = ClientNode(client_id="client_1")
    client.run_round(data_path_x="X_test.csv", data_path_y="y_test.csv")
