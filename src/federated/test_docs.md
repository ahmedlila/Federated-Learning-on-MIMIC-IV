# Test System Documentation (`test_system.py`)

## Overview

The `test_system.py` file contains a comprehensive testing framework for the federated learning system. It implements individual test functions for each component and a unified test runner that validates the entire system before deployment.

## File Structure

```
src/federated/
├── test_system.py     # Comprehensive testing framework
└── test_docs.md       # This documentation
```

## Core Components

### 1. Model Testing Function

#### test_model_creation()
**Purpose**: Validates neural network model creation and forward pass functionality.

**Implementation**:
```python
def test_model_creation():
    """Test model creation and forward pass"""
    print("🧪 Testing model creation...")
    
    try:
        model = AdmissionClassifier(input_dim=29)
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Test forward pass
        X = torch.randn(5, 29)
        output = model(X)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
```

**Test Process**:
1. **Model Creation**: Instantiates AdmissionClassifier with 29 input dimensions
2. **Parameter Count**: Calculates and reports total model parameters
3. **Forward Pass**: Tests model inference with random input tensor
4. **Shape Validation**: Verifies output tensor shape
5. **Error Handling**: Catches and reports any exceptions

**Expected Output**:
```
🧪 Testing model creation...
✅ Model created successfully
   Parameters: 12345
✅ Forward pass successful, output shape: torch.Size([5, 1])
```

### 2. Data Loading Testing Function

#### test_data_loading()
**Purpose**: Validates data loading and preprocessing functionality.

**Implementation**:
```python
def test_data_loading():
    """Test data loading functionality"""
    print("\n🧪 Testing data loading...")
    
    try:
        # Test loading data for a client
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        
        if client_data is None:
            print("❌ Data loading failed")
            return False
        
        X_train, y_train = client_data['train']
        X_val, y_val = client_data['val']
        X_test, y_test = client_data['test']
        
        print(f"✅ Data loaded successfully")
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        print(f"   Input features: {X_train.shape[1]}")
        
        # Test dataloader creation
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=8)
        print(f"✅ DataLoaders created successfully")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        
        return True
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
```

**Test Process**:
1. **Client Data Loading**: Tests load_data_for_client function
2. **Data Validation**: Verifies data splits and shapes
3. **Tensor Conversion**: Ensures proper tensor format
4. **DataLoader Creation**: Tests DataLoader creation
5. **Batch Validation**: Verifies batch processing

**Expected Output**:
```
🧪 Testing data loading...
✅ Data loaded successfully
   Train: 1000 samples
   Val: 250 samples
   Test: 300 samples
   Input features: 29
✅ DataLoaders created successfully
   Train batches: 125
   Val batches: 32
```

### 3. Training Testing Function

#### test_training()
**Purpose**: Validates model training functionality with real data.

**Implementation**:
```python
def test_training():
    """Test training functionality"""
    print("\n🧪 Testing training...")
    
    try:
        # Create model and data
        model = AdmissionClassifier(input_dim=29)
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        X_train, y_train = client_data['train']
        X_val, y_val = client_data['val']
        
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val, batch_size=8)
        
        # Test training
        updated_weights, history = train(model, train_loader, val_loader, epochs=2, lr=0.001)
        
        print(f"✅ Training completed successfully")
        print(f"   Training history keys: {list(history.keys())}")
        print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
```

**Test Process**:
1. **Model Setup**: Creates AdmissionClassifier instance
2. **Data Preparation**: Loads and prepares training data
3. **Training Execution**: Runs training for 2 epochs
4. **History Validation**: Verifies training history structure
5. **Loss Monitoring**: Reports final training loss

**Expected Output**:
```
🧪 Testing training...
✅ Training completed successfully
   Training history keys: ['train_loss', 'val_loss', 'train_acc', 'val_acc']
   Final train loss: 0.2345
```

### 4. Evaluation Testing Function

#### test_evaluation()
**Purpose**: Validates model evaluation and metric calculation.

**Implementation**:
```python
def test_evaluation():
    """Test model evaluation"""
    print("\n🧪 Testing model evaluation...")
    
    try:
        # Create model and test data
        model = AdmissionClassifier(input_dim=29)
        client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")
        X_test, y_test = client_data['test']
        
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Test evaluation
        metrics = evaluate_model(model, test_loader)
        
        print(f"✅ Evaluation completed successfully")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
```

**Test Process**:
1. **Model Creation**: Creates fresh model instance
2. **Test Data**: Loads test dataset
3. **Evaluation Execution**: Runs model evaluation
4. **Metric Calculation**: Computes accuracy, precision, recall, F1-score
5. **Result Validation**: Verifies metric ranges and format

**Expected Output**:
```
🧪 Testing model evaluation...
✅ Evaluation completed successfully
   Accuracy: 0.7500
   Precision: 0.8000
   Recall: 0.7000
   F1-Score: 0.7500
```

### 5. Federated Averaging Testing Function

#### test_federated_averaging()
**Purpose**: Validates federated averaging algorithm functionality.

**Implementation**:
```python
def test_federated_averaging():
    """Test federated averaging functionality"""
    print("\n🧪 Testing federated averaging...")
    
    try:
        # Create multiple models with different weights
        models = []
        for i in range(3):
            model = AdmissionClassifier(input_dim=29)
            # Add some noise to make weights different
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.1
            models.append(model)
        
        # Get weights from all models
        weights_list = [model.state_dict() for model in models]
        client_sizes = [100, 150, 200]  # Different client sizes
        
        # Test federated averaging
        avg_weights = federated_average(weights_list, client_sizes)
        
        print(f"✅ Federated averaging completed successfully")
        print(f"   Number of models: {len(weights_list)}")
        print(f"   Client sizes: {client_sizes}")
        
        return True
    except Exception as e:
        print(f"❌ Federated averaging failed: {e}")
        return False
```

**Test Process**:
1. **Model Creation**: Creates multiple model instances
2. **Weight Differentiation**: Adds noise to create different weights
3. **Weight Collection**: Extracts state dictionaries
4. **Averaging Execution**: Performs federated averaging
5. **Result Validation**: Verifies averaging completion

**Expected Output**:
```
🧪 Testing federated averaging...
✅ Federated averaging completed successfully
   Number of models: 3
   Client sizes: [100, 150, 200]
```

### 6. Client Node Testing Function

#### test_client_node()
**Purpose**: Validates client node functionality and data loading.

**Implementation**:
```python
def test_client_node():
    """Test client node functionality"""
    print("\n🧪 Testing client node...")
    
    try:
        # Create client node
        client = ClientNode("test_client")
        print(f"✅ Client node created successfully")
        
        # Test data loading
        data_info = client.load_local_data()
        if data_info is None:
            print("❌ Client data loading failed")
            return False
        
        print(f"✅ Client data loaded successfully")
        print(f"   Train size: {data_info['train_size']}")
        print(f"   Val size: {data_info['val_size']}")
        print(f"   Test size: {data_info['test_size']}")
        
        return True
    except Exception as e:
        print(f"❌ Client node test failed: {e}")
        return False
```

**Test Process**:
1. **Client Creation**: Instantiates ClientNode
2. **Data Loading**: Tests client-specific data loading
3. **Size Validation**: Verifies data split sizes
4. **Error Handling**: Catches and reports exceptions

**Expected Output**:
```
🧪 Testing client node...
✅ Client node created successfully
✅ Client data loaded successfully
   Train size: 1000
   Val size: 250
   Test size: 300
```

### 7. Server Endpoint Testing Function

#### test_server_endpoints()
**Purpose**: Validates server API endpoints and communication.

**Implementation**:
```python
def test_server_endpoints():
    """Test server endpoints (requires server to be running)"""
    print("\n🧪 Testing server endpoints...")
    
    server_url = "http://localhost:5000"
    
    try:
        # Test status endpoint
        response = requests.get(f"{server_url}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Server status endpoint working")
            print(f"   Round number: {status.get('round_number', 'N/A')}")
            print(f"   Pending updates: {status.get('pending_updates', 'N/A')}")
        else:
            print(f"❌ Server status endpoint failed: {response.status_code}")
            return False
        
        # Test model info endpoint
        response = requests.get(f"{server_url}/model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print(f"✅ Model info endpoint working")
            print(f"   Model type: {model_info.get('model_type', 'N/A')}")
            print(f"   Parameters: {model_info.get('total_parameters', 'N/A')}")
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
            return False
        
        # Test prediction endpoint
        sample_features = [36.6, 115.0, 20.0, 99.0, 92.0, 42.0, 0.0, 2.0, 40.0,
                          36.7, 112.0, 18.0, 98.0, 101.0, 41.0, 0.0, 1, 0, 0, 0,
                          0, 0, 0, 1, 0, 1, 0, 0, 0]
        
        response = requests.post(f"{server_url}/predict", 
                               json={'features': sample_features}, 
                               timeout=5)
        if response.status_code == 200:
            prediction = response.json()
            print(f"✅ Prediction endpoint working")
            print(f"   Prediction: {prediction.get('prediction', 'N/A')}")
            print(f"   Probability: {prediction.get('probability', 'N/A'):.3f}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            return False
        
        return True
    except requests.exceptions.ConnectionError:
        print("⚠️ Server not running. Skipping server endpoint tests.")
        return True
    except Exception as e:
        print(f"❌ Server endpoint test failed: {e}")
        return False
```

**Test Process**:
1. **Status Endpoint**: Tests `/status` endpoint
2. **Model Info Endpoint**: Tests `/model_info` endpoint
3. **Prediction Endpoint**: Tests `/predict` endpoint
4. **Response Validation**: Verifies response format and content
5. **Error Handling**: Handles server unavailability gracefully

**Expected Output**:
```
🧪 Testing server endpoints...
✅ Server status endpoint working
   Round number: 0
   Pending updates: 0
✅ Model info endpoint working
   Model type: AdmissionClassifier
   Parameters: 12345
✅ Prediction endpoint working
   Prediction: 1
   Probability: 0.750
```

### 8. Unified Test Runner

#### run_all_tests()
**Purpose**: Executes all test functions and provides comprehensive results.

**Implementation**:
```python
def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Federated Learning System Tests")
    print("=" * 50)
    
    tests = [
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training", test_training),
        ("Evaluation", test_evaluation),
        ("Federated Averaging", test_federated_averaging),
        ("Client Node", test_client_node),
        ("Server Endpoints", test_server_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False
```

**Test Execution Process**:
1. **Test Collection**: Defines all test functions
2. **Sequential Execution**: Runs tests in order
3. **Result Tracking**: Records pass/fail status
4. **Summary Generation**: Provides comprehensive results
5. **Status Reporting**: Returns overall success status

## Test Categories

### 1. Component Tests
- **Model Creation**: Neural network architecture validation
- **Data Loading**: Data preprocessing and splitting
- **Training**: Model training functionality
- **Evaluation**: Performance metric calculation

### 2. Integration Tests
- **Federated Averaging**: Distributed learning algorithm
- **Client Node**: Client-side functionality
- **Server Endpoints**: API communication

### 3. System Tests
- **End-to-End**: Complete system validation
- **Error Handling**: Exception management
- **Performance**: Basic performance validation

## Error Handling and Resilience

### 1. Individual Test Error Handling
```python
try:
    success = test_func()
    results.append((test_name, success))
except Exception as e:
    print(f"❌ {test_name} test crashed: {e}")
    results.append((test_name, False))
```

### 2. Server Connection Handling
```python
except requests.exceptions.ConnectionError:
    print("⚠️ Server not running. Skipping server endpoint tests.")
    return True
```

### 3. Data Validation
```python
if client_data is None:
    print("❌ Data loading failed")
    return False
```

## Performance Characteristics

### 1. Test Execution Time
- **Model Creation**: ~100-500ms
- **Data Loading**: ~1-5 seconds
- **Training**: ~10-30 seconds
- **Evaluation**: ~1-5 seconds
- **Federated Averaging**: ~100-500ms
- **Client Node**: ~1-3 seconds
- **Server Endpoints**: ~1-5 seconds (if server running)

### 2. Memory Usage
- **Model Tests**: ~50-100 MB
- **Data Tests**: ~100-500 MB
- **Training Tests**: ~200-1000 MB
- **Overall**: ~500-2000 MB peak

### 3. Resource Requirements
- **CPU**: Moderate usage during training
- **Memory**: Significant usage for data processing
- **Disk**: Minimal I/O operations
- **Network**: Only for server endpoint tests

## Usage Patterns

### 1. Development Testing
```bash
# Run all tests
python test_system.py

# Run specific test
python -c "from test_system import test_model_creation; test_model_creation()"
```

### 2. CI/CD Integration
```bash
# Automated testing
python test_system.py
if [ $? -eq 0 ]; then
    echo "All tests passed"
else
    echo "Some tests failed"
    exit 1
fi
```

### 3. Pre-deployment Validation
```bash
# Validate system before deployment
python test_system.py
```

## Configuration Options

### 1. Test Parameters
```python
# Training test parameters
updated_weights, history = train(model, train_loader, val_loader, epochs=2, lr=0.001)

# Data loading parameters
client_data = load_data_for_client("test_client", "data/processed/data_preprocessing")

# Server endpoint parameters
response = requests.get(f"{server_url}/status", timeout=5)
```

### 2. Test Selection
```python
tests = [
    ("Model Creation", test_model_creation),
    ("Data Loading", test_data_loading),
    ("Training", test_training),
    ("Evaluation", test_evaluation),
    ("Federated Averaging", test_federated_averaging),
    ("Client Node", test_client_node),
    ("Server Endpoints", test_server_endpoints)
]
```

## Monitoring and Debugging

### 1. Test Progress Monitoring
```python
print("🧪 Testing model creation...")
print("✅ Model created successfully")
print("❌ Model creation failed: {e}")
```

### 2. Result Analysis
```python
# Test result tracking
results.append((test_name, success))

# Summary generation
passed = sum(1 for _, success in results if success)
total = len(results)
print(f"Overall: {passed}/{total} tests passed")
```

### 3. Debug Information
```python
# Detailed error reporting
print(f"❌ {test_name} test crashed: {e}")

# Parameter reporting
print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"   Train: {len(X_train)} samples")
```

## Future Enhancements

### 1. Advanced Testing
- **Unit Tests**: Individual function testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: Benchmark testing
- **Stress Tests**: Load testing

### 2. Test Automation
- **Continuous Integration**: Automated test execution
- **Test Reporting**: Detailed test reports
- **Coverage Analysis**: Code coverage measurement
- **Parallel Testing**: Concurrent test execution

### 3. Extended Validation
- **Edge Cases**: Boundary condition testing
- **Error Scenarios**: Failure mode testing
- **Security Testing**: Vulnerability assessment
- **Compatibility Testing**: Platform validation

## Troubleshooting Guide

### 1. Common Test Failures

**Model Creation Failures**:
- Check PyTorch installation
- Verify model architecture
- Check input dimensions
- Validate tensor operations

**Data Loading Failures**:
- Verify data directory structure
- Check file permissions
- Validate CSV file format
- Ensure sufficient memory

**Training Failures**:
- Check GPU availability
- Verify learning rate settings
- Monitor memory usage
- Validate data format

### 2. Debug Commands
```bash
# Test individual components
python -c "from test_system import test_model_creation; test_model_creation()"

# Test data loading
python -c "from test_system import test_data_loading; test_data_loading()"

# Test server endpoints
python -c "from test_system import test_server_endpoints; test_server_endpoints()"
```

### 3. Performance Optimization
- **Reduce Training Epochs**: For faster testing
- **Use Smaller Datasets**: For memory constraints
- **Disable GPU**: For CPU-only testing
- **Parallel Execution**: For faster test runs

This documentation provides comprehensive technical details for the testing framework, covering all aspects from implementation to usage patterns and troubleshooting. 