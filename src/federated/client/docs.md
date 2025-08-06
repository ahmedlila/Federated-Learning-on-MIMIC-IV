# Client Directory Technical Documentation

## Overview

The `client` directory contains the federated learning client implementation that enables distributed training across multiple nodes without sharing raw data. Each client maintains its own local model, trains on local data, and contributes model updates to a central server.

## Directory Structure

```
src/federated/client/
├── node.py          # Main client implementation
├── Dockerfile       # Container configuration
└── docs.md          # This documentation
```

## Core Components

### 1. ClientNode Class (`node.py`)

The primary client implementation that handles federated learning operations.

#### Class Signature
```python
class ClientNode:
    def __init__(self, client_id, server_url='http://localhost:8080', device='cpu')
```

#### Key Attributes
- `client_id`: Unique identifier for the client
- `server_url`: Central server endpoint URL
- `device`: Computing device ('cpu' or 'cuda')
- `model`: Local neural network model (AdmissionClassifier)
- `training_history`: List of training session records

#### Initialization Process
1. Creates AdmissionClassifier with 29 input dimensions
2. Moves model to specified device (CPU/GPU)
3. Initializes training history tracking
4. Logs model parameter count

### 2. Core Methods

#### Data Management

**`load_local_data(data_dir="data/processed/data_preprocessing")`**
- **Purpose**: Loads and prepares client-specific data splits
- **Data Division Strategy**:
  - Combines pre-processed train/validation data
  - Uses client_id hash for reproducible random splits
  - Creates train (64%), validation (16%), test (20%) splits
- **Returns**: Dictionary with train_loader, val_loader, test data, and sizes
- **Dependencies**: `shared.utils.load_data_for_client()`, `shared.utils.create_dataloaders()`

**Data Flow**:
```
Raw Data → Pre-processing → Client-specific Split → PyTorch Tensors → DataLoaders
```

#### Model Synchronization

**`fetch_global_weights()`**
- **Purpose**: Downloads current global model weights from server
- **Protocol**: HTTP GET request to `/weights` endpoint
- **Data Format**: Pickle-serialized model weights
- **Error Handling**: Returns boolean success status
- **Dependencies**: `shared.utils.set_model_weights()`

**`send_updated_weights(weights, client_size)`**
- **Purpose**: Uploads trained model weights to server
- **Protocol**: HTTP POST request to `/update` endpoint
- **Payload**: JSON with pickled weights, client_size, and client_id
- **Data Format**: Base64-encoded pickle data
- **Dependencies**: `pickle.dumps()`, `requests.post()`

#### Training Operations

**`local_train(train_loader, val_loader=None, epochs=5, lr=0.001)`**
- **Purpose**: Performs local model training on client data
- **Training Process**:
  - Uses Adam optimizer with specified learning rate
  - Binary cross-entropy loss for classification
  - Early stopping on validation loss
  - Gradient clipping for stability
- **History Tracking**: Records training metrics and timestamps
- **Returns**: Updated model weights dictionary
- **Dependencies**: `shared.train.train()`

**`evaluate_local_model(test_loader)`**
- **Purpose**: Evaluates model performance on test data
- **Metrics**: Accuracy, F1-score, precision, recall
- **Returns**: Dictionary of evaluation metrics
- **Dependencies**: `shared.train.evaluate_model()`

#### Federated Learning Workflow

**`run_federated_round(epochs=5, lr=0.001)`**
- **Purpose**: Executes complete federated learning round
- **Workflow**:
  1. Load local data and create dataloaders
  2. Fetch global model weights from server
  3. Evaluate model before training
  4. Perform local training
  5. Evaluate model after training
  6. Send updated weights to server
- **Error Handling**: Returns boolean success status
- **Logging**: Comprehensive progress logging

#### Prediction Capabilities

**`predict(features)`**
- **Purpose**: Makes predictions using local model
- **Input**: Feature vector (29 dimensions)
- **Output**: Dictionary with prediction (0/1) and probability
- **Process**: Sigmoid activation on model output
- **Threshold**: 0.5 for binary classification

**`make_prediction_request(features)`**
- **Purpose**: Requests prediction from central server
- **Protocol**: HTTP POST to `/predict` endpoint
- **Returns**: Server prediction response

#### System Monitoring

**`get_server_status()`**
- **Purpose**: Retrieves current server status
- **Endpoint**: `/status`
- **Returns**: Server status dictionary or None

### 3. Simulation and Testing

**`run_client_simulation(num_clients=3, rounds=5)`**
- **Purpose**: Runs multi-client federated learning simulation
- **Process**:
  - Creates specified number of client instances
  - Runs federated rounds sequentially
  - Waits between rounds for server processing
  - Reports final server status
- **Usage**: Called when no command-line arguments provided

### 4. Command-Line Interface

**Main Execution Block**:
```python
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        client_id = sys.argv[1]
        client = ClientNode(client_id)
        client.run_federated_round()
    else:
        run_client_simulation(num_clients=3, rounds=3)
```

**Usage Patterns**:
```bash
# Run single client
python node.py client_1

# Run simulation (default)
python node.py
```

## Docker Configuration

### Dockerfile Analysis

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . /app
COPY ../shared /app/shared
RUN pip install torch requests
CMD ["python", "node.py"]
```

**Configuration Details**:
- **Base Image**: Python 3.9 slim
- **Working Directory**: `/app`
- **Dependencies**: PyTorch, requests
- **Shared Code**: Copies shared utilities to container
- **Entry Point**: Runs node.py by default

**Build Process**:
1. Installs Python 3.9 base image
2. Sets working directory to `/app`
3. Copies client code to container
4. Copies shared utilities from parent directory
5. Installs required Python packages
6. Sets default command to run client node

## Integration Points

### 1. Server Communication

**HTTP Endpoints Used**:
- `GET /weights` - Fetch global model weights
- `POST /update` - Send local model updates
- `GET /status` - Get server status
- `POST /predict` - Request server predictions

**Data Formats**:
- **Weights**: Pickle-serialized PyTorch state_dict
- **Updates**: JSON with base64-encoded weights
- **Status**: JSON response with server metrics
- **Predictions**: JSON with feature vectors

### 2. Shared Utilities Integration

**Imported Functions**:
```python
from shared.model import AdmissionClassifier
from shared.train import train, evaluate_model
from shared.utils import (
    set_model_weights, 
    get_model_weights, 
    load_data_for_client, 
    create_dataloaders
)
```

**Dependencies**:
- `shared/model.py`: Neural network architecture
- `shared/train.py`: Training and evaluation functions
- `shared/utils.py`: Data loading and model utilities

### 3. External Usage

**Demo Integration** (`demo.py`):
```python
from client.node import ClientNode

# Create client instances
client = ClientNode(f"client_{i+1}", server_url)

# Run federated rounds
success = client.run_federated_round()
```

**Testing Integration** (`test_system.py`):
```python
from client.node import ClientNode

def test_client_node():
    client = ClientNode("test_client")
    data_info = client.load_local_data()
    # Test client functionality
```

**Runner Integration** (`run_federated_learning.py`):
```python
def run_client(client_id):
    result = subprocess.run([
        sys.executable, "client/node.py", client_id
    ], cwd=os.path.dirname(os.path.abspath(__file__)))
```

## Data Flow Architecture

### 1. Initialization Flow
```
Client Creation → Model Initialization → Device Assignment → History Setup
```

### 2. Federated Round Flow
```
Load Data → Fetch Global Weights → Pre-training Evaluation → 
Local Training → Post-training Evaluation → Send Updates → 
Server Aggregation → New Global Model
```

### 3. Data Division Strategy
```
Original Dataset (100%)
├── Client 1 (80% of data)
│   ├── Training (64% of original)
│   ├── Validation (16% of original)
│   └── Test (20% of original)
├── Client 2 (80% of data)
│   ├── Training (64% of original)
│   ├── Validation (16% of original)
│   └── Test (20% of original)
└── Client N (80% of data)
    ├── Training (64% of original)
    ├── Validation (16% of original)
    └── Test (20% of original)
```

## Error Handling and Resilience

### 1. Network Communication
- **Timeout Handling**: 5-second timeouts for HTTP requests
- **Connection Errors**: Graceful degradation with error logging
- **Status Code Validation**: Checks HTTP response codes
- **Retry Logic**: Implicit retry through exception handling

### 2. Data Processing
- **File Loading**: Try-catch blocks for CSV loading
- **Tensor Conversion**: Error handling for data type conversion
- **Model Operations**: Exception handling for training/evaluation

### 3. Model Operations
- **Weight Loading**: Validation of server response format
- **Training Errors**: Graceful failure with detailed logging
- **Device Compatibility**: CPU fallback for GPU errors

## Performance Characteristics

### 1. Memory Usage
- **Model Size**: ~29 input dimensions, configurable architecture
- **Data Loading**: Batch processing with DataLoaders
- **History Storage**: Append-only list with timestamp tracking

### 2. Computational Requirements
- **Training**: Configurable epochs and learning rate
- **Device Support**: CPU and CUDA device compatibility
- **Batch Processing**: Default batch size of 16

### 3. Network Overhead
- **Weight Transfer**: Pickle serialization of model state
- **Update Frequency**: Per-round communication with server
- **Payload Size**: Model weights + metadata (~few MB)

## Security Considerations

### 1. Data Privacy
- **Local Training**: Raw data never leaves client
- **Weight Sharing**: Only model parameters shared
- **No Data Transmission**: Features and labels remain local

### 2. Communication Security
- **HTTP Protocol**: Standard web protocols
- **Data Serialization**: Pickle format for model weights
- **Error Handling**: No sensitive data in error messages

### 3. Model Protection
- **Weight Validation**: Server-side validation of received weights
- **Size Verification**: Client data size reporting
- **Update Tracking**: Server maintains update history

## Configuration Options

### 1. Client Configuration
```python
# Basic configuration
client = ClientNode(
    client_id="client_1",
    server_url="http://localhost:8080",
    device="cpu"
)

# Advanced configuration
client = ClientNode(
    client_id="client_1",
    server_url="https://federated-server.com:5000",
    device="cuda"
)
```

### 2. Training Parameters
```python
# Default training
client.local_train(train_loader, val_loader, epochs=5, lr=0.001)

# Custom training
client.local_train(
    train_loader, 
    val_loader, 
    epochs=10, 
    lr=0.0001
)
```

### 3. Data Configuration
```python
# Default data directory
data_info = client.load_local_data()

# Custom data directory
data_info = client.load_local_data("custom/data/path")
```

## Monitoring and Logging

### 1. Client Logging
- **Initialization**: Device and model parameter logging
- **Training Progress**: Epoch-by-epoch metrics
- **Communication**: Server request/response logging
- **Error Reporting**: Detailed exception information

### 2. Performance Metrics
- **Training History**: Loss and accuracy tracking
- **Evaluation Metrics**: F1-score, precision, recall
- **Timing Information**: Round completion times
- **Data Statistics**: Dataset sizes and distributions

### 3. Debug Information
- **Model State**: Parameter counts and device placement
- **Data Loading**: File paths and data shapes
- **Network Status**: Server connectivity and response times

## Future Enhancements

### 1. Scalability Improvements
- **Async Communication**: Non-blocking server requests
- **Batch Updates**: Aggregated weight updates
- **Caching**: Local model weight caching

### 2. Advanced Features
- **Differential Privacy**: Noise addition to updates
- **Secure Aggregation**: Homomorphic encryption support
- **Model Compression**: Quantization and pruning

### 3. Monitoring Enhancements
- **Real-time Metrics**: Live training progress
- **Visualization**: Training curve plotting
- **Alerting**: Performance threshold notifications

## Troubleshooting Guide

### 1. Common Issues

**Connection Errors**:
- Verify server is running on specified URL
- Check network connectivity
- Validate server endpoint availability

**Data Loading Failures**:
- Ensure data directory exists
- Verify CSV file formats
- Check file permissions

**Training Errors**:
- Validate model architecture compatibility
- Check device availability (CPU/GPU)
- Verify data tensor shapes

### 2. Debug Commands
```bash
# Test client functionality
python -c "from client.node import ClientNode; c = ClientNode('test'); print('OK')"

# Check data loading
python -c "from client.node import ClientNode; c = ClientNode('test'); print(c.load_local_data())"

# Test server communication
python -c "from client.node import ClientNode; c = ClientNode('test'); print(c.get_server_status())"
```

### 3. Log Analysis
- **Client ID**: Track specific client behavior
- **Timestamps**: Identify performance bottlenecks
- **Error Messages**: Debug communication issues
- **Metrics**: Monitor training progress

This documentation provides comprehensive technical details for the client directory implementation, covering all aspects from architecture to usage patterns and troubleshooting. 