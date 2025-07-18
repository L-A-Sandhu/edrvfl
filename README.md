# Ensemble Deep Random Vector Functional Link with Skip Connections (edRVFL-SC)

Implementation of the Ensemble Deep Random Vector Functional Link Network with Skip Connections (edRVFL-SC) based on the research paper "Stacked Ensemble Deep Random Vector Functional Link Network With Residual Learning for Medium-Scale Time-Series Forecasting".

## Key Features

- **Skip Connections:** Incorporates features from previous layers to enhance information flow.
- **Deep Ensemble Learning:** Combines predictions from multiple hidden layers.
- **Non-iterative Training:** Efficient closed-form solution for output weights.
- **Flexible Activation:** Supports ReLU, Sigmoid, Tanh, and Radial Basis activation functions.
- **Automatic Feature Concatenation:** Dynamically builds design matrices with skip connections.
- **Model Analysis:** Provides detailed parameter and FLOPs estimation.

## Installation

```bash
pip install ed-rvfl-sc
```

## Usage

### Basic Example

```python
import numpy as np
from edrvfl_sc import edRVFL_SC

# Generate sample data
X_train = np.random.rand(1000, 20)
y_train = np.random.rand(1000, 1)
X_test = np.random.rand(200, 20)

# Initialize and train model
model = edRVFL_SC(
    num_units=128,       # Number of hidden neurons per layer
    activation='relu',   # Activation function: 'relu', 'sigmoid', 'tanh', or 'radbas'
    lambda_=0.01,        # Regularization parameter
    Lmax=5,              # Number of hidden layers
    deep_boosting=0.95   # Layer scaling factor
)
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Analyze model size
params, flops = model._get_model_size()
print(f"Total parameters: {params:,}")
print(f"FLOPs per prediction: {flops:,}")
```

## Key Hyperparameters

| Parameter     | Description                                            | Default |
|---------------|--------------------------------------------------------|---------|
| num_units     | Number of hidden neurons per layer                     | 128     |
| activation    | Activation function ('relu', 'sigmoid', 'tanh', or 'radbas') | 'relu'  |
| lambda_       | Regularization coefficient                             | 0.01    |
| Lmax          | Number of hidden layers                                | 3       |
| deep_boosting | Layer scaling factor                                   | 1.0     |

## Model Architecture

The edRVFL-SC architecture features:

- **Input Layer:** Handles multi-dimensional inputs with automatic bias augmentation.
- **Hidden Layers:**
  - Randomly initialized weights fixed during training.
  - Multiple activation function options.
  - Deep boosting factor scales layer outputs.
- **Skip Connections:**
  - Layer L incorporates outputs from layer L-2.
  - Progressive feature concatenation.
- **Output Calculation:**
  - Each layer produces intermediate predictions.
  - Final prediction is ensemble average of all layer outputs.

## Advanced Features

### Input Handling

- Automatically processes 2D and 3D input tensors.
- Adds bias term to inputs.
- Flattens 3D inputs to 2D arrays.

### Skip Connection Mechanism

```python
# Build design matrix D with skip connections
if L == 0:
    D = np.concatenate([X_train, H], axis=1)
else:
    prev_features = self.H_store_train[L-2] if L >= 2 else np.empty((X_train.shape[0], 0))
    D = np.concatenate([
        self._add_bias(X_train),
        H,
        prev_features
    ], axis=1)
```

### Regularized Output Calculation

```python
# Efficient regularized least squares solution
if D.shape[0] < D.shape[1]:
    beta = D.T @ np.linalg.inv(D @ D.T + self.lambda_ * np.eye(D.shape[0])) @ Y_train
else:
    beta = np.linalg.inv(D.T @ D + self.lambda_ * np.eye(D.shape[1])) @ D.T @ Y_train
```

## Performance Analysis

The `_get_model_size()` method provides detailed computational analysis:

**Layer-wise FLOPs Breakdown:**

| Layer | Input Dim | D Features | W FLOPs | Act FLOPs | Beta FLOPs |
|-------|-----------|------------|---------|-----------|------------|
| 0     | 21        | 148        | 5,376   | 128       | 296        |
| 1     | 149       | 426        | 63,488  | 128       | 852        |
| 2     | 427       | 810        | 276,736 | 128       | 1,620      |

**Total Calculations:**

- Parameters: 116,556
- FLOPs: 349,204

## Documentation

### Class `edRVFL_SC`

#### `__init__(self, num_units, activation='relu', lambda_=0.1, Lmax=3, deep_boosting=1.0)`

Initialize the edRVFL-SC model.

**Parameters:**

- `num_units`: Number of hidden units in each layer.
- `activation`: Activation function ('relu', 'sigmoid', 'tanh', or 'radbas').
- `lambda_`: Regularization coefficient.
- `Lmax`: Number of hidden layers.
- `deep_boosting`: Scaling factor for hidden layer outputs.

#### `train(self, X_train, Y_train)`

Train the edRVFL-SC model.

**Parameters:**

- `X_train`: Input features (2D or 3D array).
- `Y_train`: Target values.

**Returns:**

- Trained model instance.

#### `predict(self, X_test)`

Make predictions using the trained model.

**Parameters:**

- `X_test`: Input features for prediction.

**Returns:**

- Model predictions.

#### `_get_model_size(self)`

Calculate and print model size information.

**Returns:**

- Tuple of (total_parameters, total_FLOPs).

## Reference

This implementation is based on the research paper:

**Stacked Ensemble Deep Random Vector Functional Link Network With Residual Learning for Medium-Scale Time-Series Forecasting**

```bibtex
@article{hu2022ensemble,
  title={Ensemble deep random vector functional link neural network for regression},
  author={Hu, Minghui and Chion, Jet Herng and Suganthan, Ponnuthurai Nagaratnam and Katuwal, Rakesh Kumar},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  volume={53},
  number={5},
  pages={2604--2615},
  year={2022},
  publisher={IEEE}
}
```

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.