import pytest
import numpy as np
from ml.model import train_model, inference, compute_model_metrics, save_model, load_model

def test_train_and_inference():
    """
    Test that the model can be trained and make predictions of correct shape and type.
    """
    X = np.array([[0, 1], [1, 0], [0, 0], [1, 1]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})

def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns values in [0, 1] for valid input.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_save_and_load_model(tmp_path):
    """
    Test that a trained model can be saved and loaded, and predictions remain the same.
    """
    X = np.array([[0, 1], [1, 0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    file_path = tmp_path / "model.pkl"
    save_model(model, file_path)
    loaded_model = load_model(file_path)
    preds_original = inference(model, X)
    preds_loaded = inference(loaded_model, X)
    assert np.array_equal(preds_original, preds_loaded)
