# Server Directory Technical Documentation

## Overview

The `server` directory contains the central federated learning server implementation that orchestrates distributed training across multiple clients. The server maintains a global model, aggregates client updates using federated averaging, and provides prediction and evaluation services.

## Directory Structure

```
src/federated/server/
├── central.py        # Main server implementation
└── docs.md           # This documentation
```

## Core Components

### 1. Flask Application (`central.py`)

The primary server implementation built on Flask framework that handles federated learning coordination.

#### Application Initialization
```python
app = Flask(__name__)

# Global variables
model = AdmissionClassifier(input_dim=29)
clients_updates = []
client_sizes = []
round_number = 0
training_history = []
model_evaluation = {}
```

#### Key Global Variables
- `model`: Global neural network model (AdmissionClassifier)
- `clients_updates`: List of model weight updates from clients
- `client_sizes`: List of client data sizes for weighted averaging
- `round_number`: Current federated learning round
- `training_history`: Historical round information
- `model_evaluation`: Model performance metrics per round

### 2. API Endpoints

#### Model Weight Distribution

**`GET /weights`**
- **Purpose**: Distributes current global model weights to clients
- **Response**: Pickle-serialized model state_dict
- **MIME Type**: `application/octet-stream`
- **Usage**: Clients fetch weights before local training
- **Implementation**:
  ```python
  @app.route('/weights', methods=['GET'])
  def send_weights():
      buffer = pickle.dumps(get_model_weights(model))
      return send_file(io.BytesIO(buffer), mimetype='application/octet-stream')
  ```

#### Client Update Aggregation

**`POST /update`**
- **Purpose**: Receives and aggregates client model updates
- **Request Format**: JSON with pickled weights, client_size, and client_id
- **Aggregation Trigger**: Configurable minimum client threshold (default: 3)
- **Process**:
  1. Receives client updates and data sizes
  2. Performs federated averaging when threshold reached
  3. Updates global model with averaged weights
  4. Evaluates global model performance
  5. Logs round information
  6. Resets for next round
- **Response**: JSON with aggregation status and round information

**Aggregation Logic**:
```python
if len(clients_updates) >= min_clients:
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
```

#### Prediction Service

**`POST /predict`**
- **Purpose**: Makes predictions using the global model
- **Request Format**: JSON with feature vector (29 dimensions)
- **Response**: JSON with prediction (0/1), probability, and features used
- **Process**:
  - Converts features to PyTorch tensor
  - Runs model inference
  - Applies sigmoid activation
  - Returns binary prediction and probability
- **Error Handling**: Returns 400 status for invalid requests

**Implementation**:
```python
@app.route('/predict', methods=['POST'])
def predict():
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
```

#### Model Evaluation

**`POST /evaluate`**
- **Purpose**: Evaluates global model on provided test data
- **Request Format**: JSON with X_test and y_test tensors
- **Response**: JSON with evaluation metrics (accuracy, F1-score, etc.)
- **Process**:
  - Creates DataLoader from provided test data
  - Runs model evaluation
  - Returns comprehensive metrics
- **Usage**: External evaluation requests

#### System Status

**`GET /status`**
- **Purpose**: Provides current server status and metrics
- **Response**: JSON with round number, pending updates, model info, and history
- **Information**:
  - Current round number
  - Number of pending client updates
  - Model parameter count
  - Training history
  - Model evaluation results

**Response Format**:
```json
{
    "round_number": 5,
    "pending_updates": 2,
    "model_parameters": 12345,
    "training_history": [...],
    "model_evaluation": {...}
}
```

#### Model Information

**`GET /model_info`**
- **Purpose**: Provides detailed model architecture information
- **Response**: JSON with model type, parameters, and structure
- **Information**:
  - Model type (AdmissionClassifier)
  - Input dimensions (29)
  - Total and trainable parameters
  - Model structure string

#### Model Reset

**`POST /reset`**
- **Purpose**: Resets global model to initial state
- **Process**:
  - Reinitializes AdmissionClassifier
  - Clears all global variables
  - Resets round number to 0
- **Usage**: System reset for new training sessions

### 3. Core Functions

#### Global Model Evaluation

**`evaluate_global_model()`**
- **Purpose**: Evaluates global model on test dataset
- **Process**:
  1. Loads test data from CSV files
  2. Creates DataLoader for batch processing
  3. Runs model evaluation
  4. Stores results with timestamp
  5. Saves evaluation to log file
- **Storage**: Results saved to `logs/evaluation_round_{N}.json`
- **Metrics**: Accuracy, F1-score, precision, recall

**Implementation**:
```python
def evaluate_global_model():
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
        
        # Save evaluation to file
        with open(f'logs/evaluation_round_{round_number}.json', 'w') as f:
            json.dump(model_evaluation[f'round_{round_number}'], f, indent=2)
    except Exception as e:
        print(f"[Server] Error evaluating model: {e}")
```

## Federated Learning Workflow

### 1. Round Initialization
```
Server starts → Global model initialized → Clients connect → 
Round begins → Clients fetch weights
```

### 2. Client Training Phase
```
Clients train locally → Send updates to server → 
Server collects updates → Check aggregation threshold
```

### 3. Model Aggregation
```
Threshold reached → Federated averaging → 
Update global model → Evaluate performance → 
Log round data → Reset for next round
```

### 4. Aggregation Algorithm

**Federated Averaging Process**:
1. **Weight Collection**: Gather model weights from all participating clients
2. **Size Weighting**: Weight each client's contribution by their data size
3. **Averaging**: Compute weighted average of model parameters
4. **Global Update**: Apply averaged weights to global model

**Mathematical Representation**:
```
w_global = Σ(client_size_i * w_i) / Σ(client_size_i)
```

## Data Flow Architecture

### 1. Weight Distribution Flow
```
Client Request → Server → Model Weights → Pickle Serialization → 
HTTP Response → Client Receives → Model Update
```

### 2. Update Aggregation Flow
```
Client Update → HTTP POST → Server Receives → 
Weight Deserialization → Storage → Threshold Check → 
Aggregation → Global Model Update → Evaluation → Logging
```

### 3. Prediction Flow
```
Feature Vector → HTTP POST → Tensor Conversion → 
Model Inference → Sigmoid Activation → 
Prediction + Probability → JSON Response
```

## Integration Points

### 1. Shared Utilities Integration

**Imported Functions**:
```python
from shared.model import AdmissionClassifier
from shared.utils import get_model_weights, set_model_weights, federated_average
from shared.train import evaluate_model as evaluate_model_func
```

**Dependencies**:
- `shared/model.py`: Neural network architecture definition
- `shared/utils.py`: Model weight management and federated averaging
- `shared/train.py`: Model evaluation functions

### 2. External Usage

**Demo Integration** (`demo.py`):
```python
# Start server in separate process
server_process = subprocess.Popen([
    sys.executable, 'server/central.py'
], cwd=os.path.dirname(os.path.abspath(__file__)))

# Check server status
response = requests.get(f"{self.server_url}/status")
```

**Runner Integration** (`run_federated_learning.py`):
```python
def run_server():
    result = subprocess.run([sys.executable, "server/central.py"],
                          cwd=os.path.dirname(os.path.abspath(__file__)))
```

**Testing Integration** (`test_system.py`):
```python
def test_server_endpoints():
    # Test status endpoint
    response = requests.get(f"{server_url}/status", timeout=5)
    
    # Test prediction endpoint
    response = requests.post(f"{server_url}/predict", 
                           json={'features': sample_features}, 
                           timeout=5)
```

**Docker Integration** (`docker-compose.yml`):
```yaml
central:
  build: ./server
  volumes:
    - ./shared:/app/shared
  ports:
    - "5000:5000"
```

## Error Handling and Resilience

### 1. Request Validation
- **JSON Parsing**: Try-catch blocks for malformed JSON
- **Data Type Validation**: Tensor conversion error handling
- **Model State**: Validation of model operations

### 2. Network Resilience
- **Connection Errors**: Graceful handling of client disconnections
- **Timeout Handling**: Configurable request timeouts
- **Partial Updates**: Handling incomplete client submissions

### 3. Model Operations
- **Weight Validation**: Verification of received model weights
- **Aggregation Errors**: Fallback for failed federated averaging
- **Evaluation Failures**: Error handling for model evaluation

### 4. File System Operations
- **Log Directory**: Automatic creation of logs directory
- **File Writing**: Error handling for evaluation log writing
- **Data Loading**: Exception handling for CSV file loading

## Performance Characteristics

### 1. Memory Management
- **Model Storage**: Single global model instance
- **Update Buffering**: Temporary storage of client updates
- **History Tracking**: Append-only training history

### 2. Computational Requirements
- **Model Size**: ~29 input dimensions, configurable architecture
- **Aggregation**: CPU-based federated averaging
- **Evaluation**: Batch processing with DataLoaders

### 3. Network Overhead
- **Weight Transfer**: Pickle serialization of model state
- **Update Frequency**: Per-round client communication
- **Payload Size**: Model weights + metadata (~few MB)

### 4. Scalability Considerations
- **Concurrent Clients**: Flask handles multiple simultaneous connections
- **Update Buffering**: Configurable minimum client threshold
- **Memory Usage**: Linear growth with number of clients

## Security Considerations

### 1. Data Privacy
- **No Raw Data**: Server never receives client data
- **Weight-Only Sharing**: Only model parameters exchanged
- **Client Anonymity**: Client IDs for tracking only

### 2. Communication Security
- **HTTP Protocol**: Standard web protocols
- **Data Serialization**: Pickle format for model weights
- **Error Handling**: No sensitive data in error messages

### 3. Model Protection
- **Weight Validation**: Verification of received model weights
- **Size Verification**: Client data size reporting
- **Update Tracking**: Server maintains update history

## Configuration Options

### 1. Server Configuration
```python
# Default configuration
app.run(host='0.0.0.0', port=5000, debug=True)

# Production configuration
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

### 2. Aggregation Parameters
```python
# Configurable minimum clients
min_clients = int(request.args.get('min_clients', 3))

# Custom aggregation threshold
# Example: POST /update?min_clients=5
```

### 3. Model Configuration
```python
# Model architecture
model = AdmissionClassifier(input_dim=29)

# Custom model parameters
# model = AdmissionClassifier(input_dim=50, hidden_dim=128)
```

## Monitoring and Logging

### 1. Server Logging
- **Round Progress**: Round number and client count logging
- **Aggregation Events**: Federated averaging completion
- **Evaluation Results**: Model performance metrics
- **Error Reporting**: Detailed exception information

### 2. Performance Metrics
- **Round Tracking**: Historical round information
- **Client Participation**: Number of clients per round
- **Model Performance**: Accuracy and F1-score tracking
- **Timing Information**: Round completion times

### 3. File-Based Logging
- **Evaluation Logs**: JSON files per round
- **Directory Structure**: `logs/evaluation_round_{N}.json`
- **Timestamp Tracking**: ISO format timestamps
- **Metrics Storage**: Comprehensive evaluation metrics

## Deployment Options

### 1. Local Development
```bash
# Direct execution
python server/central.py

# With custom port
python server/central.py --port 5001
```

### 2. Docker Deployment
```bash
# Build and run server container
docker build -t federated-server ./server
docker run -p 5000:5000 federated-server

# Using docker-compose
docker-compose up central
```

### 3. Production Deployment
```python
# Production server configuration
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
```

## API Documentation

### 1. Endpoint Summary

| Method | Endpoint | Purpose | Request Format | Response Format |
|--------|----------|---------|----------------|-----------------|
| GET | `/weights` | Get global model weights | None | Pickle binary |
| POST | `/update` | Submit client updates | JSON with weights | JSON status |
| POST | `/predict` | Make predictions | JSON with features | JSON prediction |
| POST | `/evaluate` | Evaluate model | JSON with test data | JSON metrics |
| GET | `/status` | Get server status | None | JSON status |
| GET | `/model_info` | Get model information | None | JSON model info |
| POST | `/reset` | Reset global model | None | JSON status |

### 2. Request/Response Examples

**Weight Request**:
```bash
curl -X GET http://localhost:5000/weights -o global_weights.pkl
```

**Update Submission**:
```bash
curl -X POST http://localhost:5000/update \
  -H "Content-Type: application/json" \
  -d '{"weights": "base64_encoded_weights", "client_size": 1000, "client_id": "client_1"}'
```

**Prediction Request**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [36.6, 115.0, 20.0, ...]}'
```

**Status Check**:
```bash
curl -X GET http://localhost:5000/status
```

## Troubleshooting Guide

### 1. Common Issues

**Server Startup Failures**:
- Verify port 5000 is available
- Check Python dependencies installation
- Validate shared module imports

**Client Connection Errors**:
- Ensure server is running on correct host/port
- Check network connectivity
- Verify client URL configuration

**Aggregation Failures**:
- Confirm minimum client threshold
- Check weight format compatibility
- Validate federated averaging function

### 2. Debug Commands
```bash
# Test server startup
python -c "from server.central import app; print('Server imports OK')"

# Check server status
curl -X GET http://localhost:5000/status

# Test prediction endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0] * 29}'
```

### 3. Log Analysis
- **Round Numbers**: Track federated learning progress
- **Client Counts**: Monitor participation levels
- **Evaluation Metrics**: Analyze model performance
- **Error Messages**: Debug communication issues

## Future Enhancements

### 1. Scalability Improvements
- **Async Processing**: Non-blocking client update handling
- **Load Balancing**: Multiple server instances
- **Database Integration**: Persistent storage for large-scale deployments

### 2. Advanced Features
- **Secure Aggregation**: Homomorphic encryption support
- **Differential Privacy**: Noise addition to updates
- **Model Compression**: Weight quantization and pruning

### 3. Monitoring Enhancements
- **Real-time Metrics**: Live performance monitoring
- **Visualization**: Training progress dashboards
- **Alerting**: Performance threshold notifications

### 4. Security Enhancements
- **Authentication**: Client authentication mechanisms
- **Encryption**: End-to-end encryption for weight transfers
- **Audit Logging**: Comprehensive security event tracking

This documentation provides comprehensive technical details for the server directory implementation, covering all aspects from API design to deployment and troubleshooting. 