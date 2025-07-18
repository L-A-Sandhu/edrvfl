import numpy as np
from ed_rvfl_sc import edRVFL_SC  # Changed import

def test_train_predict():
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    
    model = edRVFL_SC(num_units=20, Lmax=2)
    model.train(X_train, y_train)
    
    X_test = np.random.rand(10, 10)
    preds = model.predict(X_test)
    
    assert preds.shape == (10, 1)
    assert not np.isnan(preds).any()