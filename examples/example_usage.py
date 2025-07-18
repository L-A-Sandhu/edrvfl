import numpy as np
from edrvfl_sc import edRVFL_SC

# Sample data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
X_test = np.random.rand(20, 10)

# Initialize and train model
model = edRVFL_SC(
    num_units=51,
    Lmax=3,
    lambda_=0.489,
    activation='relu'
)
model.train(X_train, y_train)

# Make predictions
preds = model.predict(X_test)
print("Predictions shape:", preds.shape)
