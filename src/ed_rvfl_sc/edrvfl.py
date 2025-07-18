import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import json

class edRVFL_SC:
    def __init__(self, num_units, activation='relu', lambda_=0.1, Lmax=3, deep_boosting=1.0):
        self.num_units = num_units
        self.activation = activation
        self.lambda_ = lambda_
        self.Lmax = Lmax
        self.db = deep_boosting
        self.weight_array = []
        self.beta_computed = []
        self.H_store_train = []
        self.H_store_test = []
        self.input_shape = None
        self.dtype = np.float32
        
        # Dimension tracking
        self.A_features_list = []  # Input dimension to W at each layer
        self.D_features_list = []  # Feature dimension of D at each layer
        self.output_dim = None

    def _activate(self, x):
        if self.activation == 'relu':
            result = np.maximum(0, x)
        elif self.activation == 'sigmoid':
            x_clipped = np.clip(x, -100, 100)
            result = 1 / (1 + np.exp(-x_clipped))
        elif self.activation == 'tanh':
            result = np.tanh(x)
        elif self.activation == 'radbas':
            result = np.exp(-x ** 2)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return result.astype(self.dtype)

    def _add_bias(self, X):
        if X.ndim == 3:
            ones = np.ones((*X.shape[:-1], 1), dtype=self.dtype)
        else:
            ones = np.ones((X.shape[0], 1), dtype=self.dtype)
        return np.concatenate([X.astype(self.dtype), ones], axis=-1)

    def _reshape_input(self, X):
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1).astype(self.dtype)
        return X.astype(self.dtype)

    def train(self, X_train, Y_train):
        self.input_shape = X_train.shape
        X_flat = self._reshape_input(X_train)
        X_train = X_flat
        Y_train = Y_train.astype(self.dtype)
        self.output_dim = Y_train.shape[1]
        
        A = self._add_bias(X_train)
        self.H_store_train = []
        self.weight_array = []
        self.beta_computed = []
        self.A_features_list = []
        self.D_features_list = []
        
        for L in range(self.Lmax):
            # Track input dimension to W
            current_A_features = A.shape[1]
            self.A_features_list.append(current_A_features)
            
            # Initialize weights
            W = np.random.randn(current_A_features, self.num_units).astype(self.dtype)
            self.weight_array.append(W)
            
            # Compute hidden layer
            H = self._activate(A @ W)
            H *= self.db if L == 0 else self.db/(self.Lmax - L + 1)
            self.H_store_train.append(H)
            
            # Build design matrix D
            if L == 0:
                D = np.concatenate([X_train, H], axis=1)
            else:
                prev_features = self.H_store_train[L-2] if L >= 2 else np.empty((X_train.shape[0], 0), dtype=self.dtype)
                D = np.concatenate([
                    self._add_bias(X_train),
                    H,
                    prev_features
                ], axis=1)
            
            # Track D's feature dimension
            current_D_features = D.shape[1]
            self.D_features_list.append(current_D_features)
            
            # Compute output weights
            if D.shape[0] < D.shape[1]:
                beta = D.T @ np.linalg.inv(D @ D.T + self.lambda_ * np.eye(D.shape[0], dtype=self.dtype)) @ Y_train
            else:
                beta = np.linalg.inv(D.T @ D + self.lambda_ * np.eye(D.shape[1], dtype=self.dtype)) @ D.T @ Y_train
            self.beta_computed.append(beta.astype(self.dtype))
            
            # Update A for next layer
            if L > 1:
                A = np.concatenate([
                    self._add_bias(X_train),
                    H,
                    self.H_store_train[L-2]
                ], axis=1)
            else:
                A = np.concatenate([self._add_bias(X_train), H], axis=1)
        
        return self

    def _get_model_size(self):
        if not self.beta_computed:
            return 0, 0
        
        total_params = 0
        total_flops = 0
        activation_cost = 4 if self.activation == 'sigmoid' else 1  # 4 FLOPs for sigmoid
        
        print("\nLayer-wise FLOPs Breakdown:")
        print(f"{'Layer':<6} {'Input Dim':<10} {'D Features':<10} {'W FLOPs':<12} {'Act FLOPs':<12} {'Beta FLOPs':<12}")
        
        for L in range(self.Lmax):
            A_features = self.A_features_list[L]
            D_features = self.D_features_list[L]
            num_units = self.num_units
            
            # W matrix multiplication
            w_flops = 2 * A_features * num_units
            
            # Activation function
            act_flops = num_units * activation_cost
            
            # Beta projection
            beta_flops = 2 * D_features * self.output_dim
            
            layer_flops = w_flops + act_flops + beta_flops
            total_flops += layer_flops
            
            # Parameters count
            total_params += (A_features * num_units) + (D_features * self.output_dim)
            
            print(f"{L:<6} {A_features:<10} {D_features:<10} "
                  f"{w_flops:<12,} {act_flops:<12,} {beta_flops:<12,}")

        print("\nTotal Calculations:")
        print(f"Parameters: {total_params:,}")
        print(f"FLOPs: {total_flops:,}")
        
        return total_params, int(total_flops)

    def predict(self, X_test):
        X_flat = self._reshape_input(X_test)
        X_test = X_flat
        A_test = self._add_bias(X_test)
        self.H_store_test = []
        predictions = []
        
        for L in range(self.Lmax):
            W = self.weight_array[L]
            H = self._activate(A_test @ W)
            H *= self.db if L == 0 else self.db/(self.Lmax - L + 1)
            self.H_store_test.append(H)
            
            if L == 0:
                D_test = np.concatenate([X_test, H], axis=1)
            else:
                prev_features = self.H_store_test[L-2] if L >=2 else np.empty((X_test.shape[0], 0), dtype=self.dtype)
                D_test = np.concatenate([
                    self._add_bias(X_test),
                    self.H_store_test[L],
                    prev_features
                ], axis=1)
            
            predictions.append(D_test @ self.beta_computed[L])
            
            if L > 1:
                A_test = np.concatenate([
                    self._add_bias(X_test),
                    H,
                    self.H_store_test[L-2]
                ], axis=1)
            else:
                A_test = np.concatenate([self._add_bias(X_test), H], axis=1)

        return np.mean(predictions, axis=0)