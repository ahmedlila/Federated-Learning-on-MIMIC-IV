from flask import Flask, request, send_file
import pickle, io
import torch
from shared.model import AdmissionClassifier
from shared.utils import get_model_weights, set_model_weights

app = Flask(__name__)
model = AdmissionClassifier(input_dim=20)
clients_updates = []

@app.route('/weights', methods=['GET'])
def send_weights():
    buffer = pickle.dumps(get_model_weights(model))
    return send_file(io.BytesIO(buffer), mimetype='application/octet-stream')

@app.route('/update', methods=['POST'])
def receive_update():
    global clients_updates, model
    weights = pickle.loads(request.data)
    clients_updates.append(weights)

    if len(clients_updates) == 3:
        avg_weights = average_weights(clients_updates)
        set_model_weights(model, avg_weights)
        clients_updates = []

    return 'OK'

def average_weights(weights_list):
    avg = {}
    for k in weights_list[0].keys():
        avg[k] = sum([w[k] for w in weights_list]) / len(weights_list)
    return avg

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
