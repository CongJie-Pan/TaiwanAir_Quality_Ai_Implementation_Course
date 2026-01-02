"""
Unit Tests for Linear Regression Model

Run with:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python -m pytest src/main/python/models/test_linear_regression.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main.python.models.linear_regression import AQILinearRegression, create_scatter_plot, save_metrics_json


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'pm2.5_lag1': np.random.uniform(10, 100, n_samples),
        'pm10_lag1': np.random.uniform(20, 150, n_samples),
        'o3_lag1': np.random.uniform(5, 80, n_samples),
        'windspeed': np.random.uniform(0, 10, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    })
    
    # Synthetic AQI with known relationships
    y = pd.Series(
        25 + 0.4 * X['pm2.5_lag1'] + 0.2 * X['pm10_lag1'] - 2 * X['windspeed'] 
        + np.random.normal(0, 5, n_samples)
    )
    
    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Create a pre-trained model."""
    X, y = sample_data
    model = AQILinearRegression(standardize=False)
    model.train(X, y)
    return model


# ============================================================================
# Test: Model Initialization
# ============================================================================

class TestModelInit:
    """Tests for model initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = AQILinearRegression()
        assert model.standardize == False
        assert model.is_trained == False
        assert model.feature_names == []
    
    def test_init_with_standardize(self):
        """Test initialization with standardization."""
        model = AQILinearRegression(standardize=True)
        assert model.standardize == True


# ============================================================================
# Test: Training
# ============================================================================

class TestTraining:
    """Tests for model training."""
    
    def test_train_basic(self, sample_data):
        """Test basic training works."""
        X, y = sample_data
        model = AQILinearRegression()
        metrics = model.train(X, y)
        
        assert model.is_trained == True
        assert len(model.feature_names) == len(X.columns)
        assert 'train_r2' in metrics
        assert 'train_rmse' in metrics
        assert 'train_mae' in metrics
    
    def test_train_with_validation(self, sample_data):
        """Test training with validation data."""
        X, y = sample_data
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        model = AQILinearRegression()
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        assert 'val_r2' in metrics
        assert 'val_rmse' in metrics
        assert 'val_mae' in metrics
    
    def test_train_with_standardization(self, sample_data):
        """Test training with standardization."""
        X, y = sample_data
        model = AQILinearRegression(standardize=True)
        metrics = model.train(X, y)
        
        assert model.scaler is not None
        assert model.is_trained == True
    
    def test_metrics_valid_range(self, sample_data):
        """Test that metrics are in valid ranges."""
        X, y = sample_data
        model = AQILinearRegression()
        metrics = model.train(X, y)
        
        # R² should be between 0 and 1 for reasonable data
        assert 0 <= metrics['train_r2'] <= 1
        # RMSE and MAE should be non-negative
        assert metrics['train_rmse'] >= 0
        assert metrics['train_mae'] >= 0


# ============================================================================
# Test: Prediction
# ============================================================================

class TestPrediction:
    """Tests for model prediction."""
    
    def test_predict_shape(self, trained_model, sample_data):
        """Test prediction output shape."""
        X, _ = sample_data
        predictions = trained_model.predict(X)
        
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_not_trained_raises(self, sample_data):
        """Test that prediction on untrained model raises error."""
        X, _ = sample_data
        model = AQILinearRegression()
        
        with pytest.raises(RuntimeError):
            model.predict(X)
    
    def test_predict_values_reasonable(self, trained_model, sample_data):
        """Test prediction values are reasonable."""
        X, y = sample_data
        predictions = trained_model.predict(X)
        
        # Predictions should be in a reasonable range (not extreme)
        assert predictions.min() > -100
        assert predictions.max() < 500


# ============================================================================
# Test: Evaluation
# ============================================================================

class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_keys(self, trained_model, sample_data):
        """Test evaluation returns expected keys."""
        X, y = sample_data
        metrics = trained_model.evaluate(X, y)
        
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'n_samples' in metrics
    
    def test_evaluate_n_samples(self, trained_model, sample_data):
        """Test n_samples is correct."""
        X, y = sample_data
        metrics = trained_model.evaluate(X, y)
        
        assert metrics['n_samples'] == len(y)


# ============================================================================
# Test: Coefficients
# ============================================================================

class TestCoefficients:
    """Tests for coefficient extraction."""
    
    def test_get_coefficients_shape(self, trained_model):
        """Test coefficient table has correct number of rows."""
        coef_df = trained_model.get_coefficients()
        
        # Should have n_features + 1 (intercept)
        expected_rows = len(trained_model.feature_names) + 1
        assert len(coef_df) == expected_rows
    
    def test_get_coefficients_columns(self, trained_model):
        """Test coefficient table has expected columns."""
        coef_df = trained_model.get_coefficients()
        
        assert 'feature' in coef_df.columns
        assert 'coefficient' in coef_df.columns
        assert 'abs_coefficient' in coef_df.columns
        assert 'interpretation' in coef_df.columns
    
    def test_intercept_included(self, trained_model):
        """Test that intercept is included."""
        coef_df = trained_model.get_coefficients()
        
        assert '(intercept)' in coef_df['feature'].values
    
    def test_windspeed_negative_coefficient(self, sample_data):
        """Test that windspeed has negative coefficient (expected relationship)."""
        X, y = sample_data
        model = AQILinearRegression()
        model.train(X, y)
        
        coef_df = model.get_coefficients()
        windspeed_coef = coef_df[coef_df['feature'] == 'windspeed']['coefficient'].values[0]
        
        # Windspeed should have negative effect on AQI (higher wind = lower AQI)
        assert windspeed_coef < 0


# ============================================================================
# Test: Model Persistence
# ============================================================================

class TestPersistence:
    """Tests for save/load functionality."""
    
    def test_save_and_load(self, trained_model, sample_data):
        """Test model can be saved and loaded."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            # Save
            trained_model.save_model(save_path)
            assert (save_path.with_suffix('.joblib')).exists()
            
            # Load
            loaded_model = AQILinearRegression.load_model(save_path)
            assert loaded_model.is_trained == True
            assert loaded_model.feature_names == trained_model.feature_names
    
    def test_loaded_model_predicts_same(self, trained_model, sample_data):
        """Test loaded model gives same predictions."""
        X, _ = sample_data
        original_pred = trained_model.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            trained_model.save_model(save_path)
            loaded_model = AQILinearRegression.load_model(save_path)
            
            loaded_pred = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


# ============================================================================
# Test: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_save_metrics_json(self):
        """Test saving metrics to JSON."""
        metrics = {'r2': 0.85, 'rmse': 10.5, 'mae': 8.2}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            save_metrics_json(metrics, path)
            
            assert path.exists()
            
            import json
            with open(path) as f:
                loaded = json.load(f)
            
            assert loaded['r2'] == 0.85
    
    def test_create_scatter_plot(self, trained_model, sample_data):
        """Test scatter plot creation."""
        X, y = sample_data
        predictions = trained_model.predict(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scatter.png"
            create_scatter_plot(y.values, predictions, save_path=path)
            
            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
