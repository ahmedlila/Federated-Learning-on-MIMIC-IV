# Model Documentation (`model.py`)

## Overview

The `model.py` file contains the neural network architecture definition for the federated learning system. It implements the `AdmissionClassifier` class, a PyTorch-based neural network designed for binary classification of hospital admission predictions.

## File Structure

```
src/federated/shared/
├── model.py           # Neural network architecture
└── model_docs.md      # This documentation
```

## Core Components

### 1. AdmissionClassifier Class

The primary neural network model for hospital admission prediction.

#### Class Signature
```python
class AdmissionClassifier(nn.Module):
    def __init__(self, input_dim=29):
```

#### Architecture Overview

The model implements a feedforward neural network with the following characteristics:

- **Input Layer**: Configurable input dimensions (default: 29 features)
- **Hidden Layers**: 3 fully connected layers with decreasing dimensions
- **Activation Functions**: ReLU activation for non-linearity
- **Regularization**: Batch normalization and dropout for generalization
- **Output Layer**: Single neuron for binary classification

#### Layer Architecture

```python
self.model = nn.Sequential(
    nn.Linear(input_dim, 128),      # Input → 128 neurons
    nn.ReLU(),                      # ReLU activation
    nn.BatchNorm1d(128),           # Batch normalization
    nn.Dropout(0.3),               # 30% dropout
    nn.Linear(128, 64),            # 128 → 64 neurons
    nn.ReLU(),                     # ReLU activation
    nn.BatchNorm1d(64),           # Batch normalization
    nn.Dropout(0.2),              # 20% dropout
    nn.Linear(64, 32),            # 64 → 32 neurons
    nn.ReLU(),                    # ReLU activation
    nn.Linear(32, 1)              # 32 → 1 neuron (output)
)
```

#### Layer Details

**Layer 1: Input Processing**
- **Type**: Linear transformation
- **Dimensions**: `input_dim → 128`
- **Purpose**: Initial feature processing and dimensionality expansion
- **Parameters**: `input_dim × 128 + 128` (weights + biases)

**Layer 2: First Hidden Layer**
- **Type**: Linear transformation with regularization
- **Dimensions**: `128 → 64`
- **Activation**: ReLU (Rectified Linear Unit)
- **Regularization**: BatchNorm1d + Dropout(0.3)
- **Purpose**: Feature extraction and dimensionality reduction
- **Parameters**: `128 × 64 + 64` (weights + biases)

**Layer 3: Second Hidden Layer**
- **Type**: Linear transformation with regularization
- **Dimensions**: `64 → 32`
- **Activation**: ReLU
- **Regularization**: BatchNorm1d + Dropout(0.2)
- **Purpose**: Further feature refinement
- **Parameters**: `64 × 32 + 32` (weights + biases)

**Layer 4: Output Layer**
- **Type**: Linear transformation
- **Dimensions**: `32 → 1`
- **Activation**: None (raw logits)
- **Purpose**: Binary classification output
- **Parameters**: `32 × 1 + 1` (weights + biases)

### 2. Forward Pass Implementation

#### Method Signature
```python
def forward(self, x):
    return self.model(x)
```

#### Process Flow
1. **Input Validation**: Expects tensor of shape `(batch_size, input_dim)`
2. **Sequential Processing**: Passes through all layers in sequence
3. **Output Generation**: Returns raw logits of shape `(batch_size, 1)`

#### Data Flow
```
Input Tensor (batch_size, input_dim)
    ↓
Linear(29, 128) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Linear(128, 64) + ReLU + BatchNorm + Dropout(0.2)
    ↓
Linear(64, 32) + ReLU
    ↓
Linear(32, 1)
    ↓
Output Logits (batch_size, 1)
```

## Model Characteristics

### 1. Parameter Count

**Total Parameters**: Approximately 12,000-15,000 parameters
- **Layer 1**: `29 × 128 + 128 = 3,840` parameters
- **Layer 2**: `128 × 64 + 64 = 8,256` parameters
- **Layer 3**: `64 × 32 + 32 = 2,080` parameters
- **Layer 4**: `32 × 1 + 1 = 33` parameters
- **BatchNorm layers**: Additional parameters for normalization

### 2. Memory Requirements

**Model Size**: ~50-60 KB (uncompressed)
- **Parameters**: ~12K-15K float32 values
- **Gradients**: Same size as parameters during training
- **Activations**: Varies with batch size and input dimensions

### 3. Computational Complexity

**Forward Pass**: O(input_dim × 128 + 128 × 64 + 64 × 32 + 32 × 1)
- **Time Complexity**: Linear with input dimensions
- **Space Complexity**: Constant with respect to input size
- **Batch Processing**: Efficient parallel computation

## Design Rationale

### 1. Architecture Choices

**Input Dimension (29)**:
- Matches hospital admission feature set
- Includes vital signs, demographics, and clinical indicators
- Configurable for different feature sets

**Hidden Layer Sizes (128 → 64 → 32)**:
- Gradual dimensionality reduction
- Preserves important features while reducing complexity
- Balances expressiveness with computational efficiency

**Activation Functions**:
- **ReLU**: Prevents vanishing gradients, computationally efficient
- **No activation on output**: Raw logits for binary cross-entropy loss

### 2. Regularization Strategy

**Batch Normalization**:
- Stabilizes training process
- Reduces internal covariate shift
- Improves convergence speed

**Dropout**:
- **Layer 1**: 30% dropout (higher for larger layer)
- **Layer 2**: 20% dropout (moderate regularization)
- **Output**: No dropout (preserves final predictions)

### 3. Training Considerations

**Loss Function Compatibility**:
- Designed for `BCEWithLogitsLoss`
- Raw logits output enables stable training
- Sigmoid activation applied during inference

**Gradient Flow**:
- ReLU activations prevent vanishing gradients
- Batch normalization stabilizes gradient flow
- Appropriate layer sizes for gradient propagation

## Usage Patterns

### 1. Basic Model Creation

```python
from shared.model import AdmissionClassifier

# Default model (29 input features)
model = AdmissionClassifier()

# Custom input dimensions
model = AdmissionClassifier(input_dim=50)
```

### 2. Model Training Setup

```python
import torch

# Create model
model = AdmissionClassifier(input_dim=29)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Prepare for training
model.train()
```

### 3. Model Inference

```python
# Prepare input data
X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Set to evaluation mode
model.eval()

# Forward pass
with torch.no_grad():
    logits = model(X)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).int()
```

### 4. Federated Learning Integration

```python
# Model weight management
from shared.utils import get_model_weights, set_model_weights

# Get current weights
weights = get_model_weights(model)

# Set new weights
set_model_weights(model, new_weights)
```

## Integration Points

### 1. Training Integration (`train.py`)

```python
from shared.model import AdmissionClassifier
from shared.train import train

# Create and train model
model = AdmissionClassifier(input_dim=29)
updated_weights, history = train(model, train_loader, val_loader)
```

### 2. Client Integration (`client/node.py`)

```python
from shared.model import AdmissionClassifier

class ClientNode:
    def __init__(self, client_id):
        self.model = AdmissionClassifier(input_dim=29).to(self.device)
```

### 3. Server Integration (`server/central.py`)

```python
from shared.model import AdmissionClassifier

# Global model initialization
model = AdmissionClassifier(input_dim=29)
```

## Performance Characteristics

### 1. Training Performance

**Convergence**: Typically 10-50 epochs for convergence
- **Early stopping**: Implemented in training loop
- **Learning rate**: 0.001 with adaptive scheduling
- **Batch size**: 16-32 for optimal performance

**Memory Usage**:
- **Model parameters**: ~12K-15K parameters
- **Gradient storage**: Same size as parameters
- **Activation memory**: Varies with batch size

### 2. Inference Performance

**Speed**: ~1-10 ms per prediction (CPU)
- **Batch processing**: Efficient parallel inference
- **Memory efficient**: Minimal memory footprint
- **Scalable**: Linear scaling with batch size

### 3. Model Size

**Parameter Count**: ~12,000-15,000 parameters
- **Storage**: ~50-60 KB (uncompressed)
- **Transfer**: Efficient for federated learning
- **Compression**: Can be quantized for deployment

## Configuration Options

### 1. Input Dimension Customization

```python
# Standard hospital admission model
model = AdmissionClassifier(input_dim=29)

# Extended feature set
model = AdmissionClassifier(input_dim=50)

# Minimal feature set
model = AdmissionClassifier(input_dim=15)
```

### 2. Architecture Modifications

**Custom Layer Sizes**:
```python
# Modify layer dimensions
class CustomAdmissionClassifier(nn.Module):
    def __init__(self, input_dim=29, hidden_dims=[128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
```

### 3. Regularization Adjustments

```python
# Custom dropout rates
class AdmissionClassifier(nn.Module):
    def __init__(self, input_dim=29, dropout_rates=[0.3, 0.2, 0.1]):
        # Implementation with custom dropout rates
```

## Error Handling and Validation

### 1. Input Validation

**Shape Requirements**:
- Input tensor must be 2D: `(batch_size, input_dim)`
- Input dimension must match model configuration
- Data type should be `torch.float32`

**Validation Examples**:
```python
# Correct input shape
X = torch.randn(32, 29)  # batch_size=32, features=29

# Incorrect input shape
X = torch.randn(29)      # Missing batch dimension
X = torch.randn(32, 30)  # Wrong feature dimension
```

### 2. Device Compatibility

**CPU/GPU Support**:
```python
# Automatic device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Explicit device assignment
model = model.to('cpu')
model = model.to('cuda:0')
```

## Testing and Validation

### 1. Model Creation Test

```python
def test_model_creation():
    model = AdmissionClassifier(input_dim=29)
    assert model is not None
    assert sum(p.numel() for p in model.parameters()) > 0
```

### 2. Forward Pass Test

```python
def test_forward_pass():
    model = AdmissionClassifier(input_dim=29)
    X = torch.randn(10, 29)  # 10 samples, 29 features
    output = model(X)
    assert output.shape == (10, 1)
```

### 3. Parameter Count Test

```python
def test_parameter_count():
    model = AdmissionClassifier(input_dim=29)
    param_count = sum(p.numel() for p in model.parameters())
    assert 10000 < param_count < 20000  # Expected range
```

## Future Enhancements

### 1. Architecture Improvements

**Attention Mechanisms**:
- Self-attention layers for feature interaction
- Multi-head attention for complex patterns
- Transformer-based architecture

**Residual Connections**:
- Skip connections for better gradient flow
- ResNet-style architecture
- Improved training stability

### 2. Regularization Enhancements

**Advanced Regularization**:
- Layer normalization
- Weight decay optimization
- Label smoothing

**Dropout Variations**:
- Spatial dropout
- Alpha dropout
- Adaptive dropout rates

### 3. Performance Optimizations

**Model Compression**:
- Weight quantization
- Pruning techniques
- Knowledge distillation

**Efficient Inference**:
- Model optimization
- TensorRT integration
- ONNX export support

## Troubleshooting Guide

### 1. Common Issues

**Shape Mismatch Errors**:
- Verify input tensor dimensions
- Check model input_dim parameter
- Ensure batch dimension is present

**Memory Issues**:
- Reduce batch size
- Use gradient checkpointing
- Enable mixed precision training

**Training Instability**:
- Adjust learning rate
- Modify dropout rates
- Check data normalization

### 2. Debug Commands

```python
# Test model creation
python -c "from shared.model import AdmissionClassifier; m = AdmissionClassifier(); print('Model created successfully')"

# Check parameter count
python -c "from shared.model import AdmissionClassifier; m = AdmissionClassifier(); print(f'Parameters: {sum(p.numel() for p in m.parameters())}')"

# Test forward pass
python -c "from shared.model import AdmissionClassifier; import torch; m = AdmissionClassifier(); x = torch.randn(1, 29); print(m(x).shape)"
```

### 3. Performance Monitoring

**Training Metrics**:
- Loss convergence curves
- Accuracy progression
- Gradient norm monitoring

**Memory Usage**:
- Parameter count tracking
- Activation memory monitoring
- GPU memory utilization

This documentation provides comprehensive technical details for the model architecture, covering all aspects from design rationale to usage patterns and troubleshooting. 