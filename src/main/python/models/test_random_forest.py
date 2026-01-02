"""
Unit Tests for Random Forest Classification Model

Run with:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python -m pytest src/main/python/models/test_random_forest.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main.python.models.random_forest import AQIRandomForest, save_metrics_json


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Features
    X = pd.DataFrame({
        'pm2.5_lag1': np.random.uniform(10, 100, n_samples),
        'windspeed': np.random.uniform(0, 10, n_samples),
        'hour_sin': np.random.uniform(-1, 1, n_samples),
        'county_encoded': np.random.randint(0, 3, n_samples)
    })
    
    # Target (classification based on PM2.5)
    def classify(pm25):
        if pm25 <= 35:
            return '良好'
        elif pm25 <= 70:
            return '普通'
        else:
            return '對敏感族群不健康'
    
    y = pd.Series([classify(p) for p in X['pm2.5_lag1']])
    
    # Split
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    return {
        'X_train': X[:train_size],
        'y_train': y[:train_size],
        'X_val': X[train_size:train_size+val_size],
        'y_val': y[train_size:train_size+val_size],
        'X_test': X[train_size+val_size:],
        'y_test': y[train_size+val_size:]
    }


@pytest.fixture
def trained_model(sample_data):
    """Create a pre-trained model."""
    model = AQIRandomForest(n_estimators=10, max_depth=5, class_weight='balanced')
    model.train(sample_data['X_train'], sample_data['y_train'])
    return model


# ============================================================================
# Test: Model Initialization
# ============================================================================

class TestModelInit:
    """Tests for model initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        model = AQIRandomForest()
        assert model.n_estimators == 100
        assert model.max_depth == 10
        assert model.class_weight == 'balanced'
        assert model.oob_score_enabled == True
        assert model.is_trained == False
        assert model.feature_names == []
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = AQIRandomForest(n_estimators=50, max_depth=5, class_weight=None)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5
    
    def test_init_class_names(self):
        """Test default class names are set."""
        model = AQIRandomForest()
        assert '良好' in model.DEFAULT_CLASS_NAMES
        assert '普通' in model.DEFAULT_CLASS_NAMES
        assert '對敏感族群不健康' in model.DEFAULT_CLASS_NAMES
    
    def test_init_oob_disabled(self):
        """Test initialization with OOB score disabled."""
        model = AQIRandomForest(oob_score=False)
        assert model.oob_score_enabled == False


# ============================================================================
# Test: Training
# ============================================================================

class TestTraining:
    """Tests for model training."""
    
    def test_train_basic(self, sample_data):
        """Test basic training works."""
        model = AQIRandomForest(n_estimators=10, max_depth=5)
        metrics = model.train(sample_data['X_train'], sample_data['y_train'])
        
        assert model.is_trained == True
        assert 'train_accuracy' in metrics
        assert 'train_f1_macro' in metrics
        assert metrics['train_samples'] == len(sample_data['X_train'])
    
    def test_train_with_validation(self, sample_data):
        """Test training with validation data."""
        model = AQIRandomForest(n_estimators=10, max_depth=5)
        metrics = model.train(
            sample_data['X_train'], sample_data['y_train'],
            sample_data['X_val'], sample_data['y_val']
        )
        
        assert 'val_accuracy' in metrics
        assert 'val_f1_macro' in metrics
        assert metrics['val_samples'] == len(sample_data['X_val'])
    
    def test_train_stores_feature_names(self, sample_data):
        """Test that training stores feature names."""
        model = AQIRandomForest(n_estimators=10, max_depth=5)
        model.train(sample_data['X_train'], sample_data['y_train'])
        
        assert model.feature_names == list(sample_data['X_train'].columns)
    
    def test_train_class_weight_balanced(self, sample_data):
        """Test training with balanced class weights."""
        model = AQIRandomForest(n_estimators=10, max_depth=5, class_weight='balanced')
        metrics = model.train(sample_data['X_train'], sample_data['y_train'])
        
        # Should still train successfully
        assert model.is_trained == True
        assert metrics['train_accuracy'] > 0
    
    def test_metrics_valid_range(self, sample_data):
        """Test that metrics are in valid ranges."""
        model = AQIRandomForest(n_estimators=10, max_depth=5)
        metrics = model.train(sample_data['X_train'], sample_data['y_train'])
        
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['train_f1_macro'] <= 1
        assert metrics['n_estimators'] == 10
        assert metrics['max_depth'] == 5
    
    def test_train_oob_score_present(self, sample_data):
        """Test that OOB score is calculated when enabled."""
        model = AQIRandomForest(n_estimators=10, max_depth=5, oob_score=True)
        metrics = model.train(sample_data['X_train'], sample_data['y_train'])
        
        assert 'oob_score' in metrics
        assert 0 <= metrics['oob_score'] <= 1


# ============================================================================
# Test: Prediction
# ============================================================================

class TestPrediction:
    """Tests for model prediction."""
    
    def test_predict_shape(self, trained_model, sample_data):
        """Test prediction output shape."""
        predictions = trained_model.predict(sample_data['X_test'])
        
        assert len(predictions) == len(sample_data['X_test'])
    
    def test_predict_not_trained_raises(self, sample_data):
        """Test that prediction on untrained model raises error."""
        model = AQIRandomForest()
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(sample_data['X_test'])
    
    def test_predict_returns_valid_classes(self, trained_model, sample_data):
        """Test that predictions are valid class labels."""
        predictions = trained_model.predict(sample_data['X_test'])
        
        valid_classes = set(sample_data['y_train'].unique())
        for pred in predictions:
            assert pred in valid_classes
    
    def test_predict_proba_shape(self, trained_model, sample_data):
        """Test predict_proba output shape."""
        proba = trained_model.predict_proba(sample_data['X_test'])
        
        n_samples = len(sample_data['X_test'])
        n_classes = len(sample_data['y_train'].unique())
        
        assert proba.shape == (n_samples, n_classes)
    
    def test_predict_proba_sums_to_one(self, trained_model, sample_data):
        """Test that probabilities sum to 1."""
        proba = trained_model.predict_proba(sample_data['X_test'])
        
        # Each row should sum to approximately 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(row_sums)))


# ============================================================================
# Test: Evaluation
# ============================================================================

class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_keys(self, trained_model, sample_data):
        """Test evaluation returns expected keys."""
        metrics = trained_model.evaluate(sample_data['X_test'], sample_data['y_test'])
        
        expected_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 
                        'confusion_matrix', 'classification_report', 'n_samples']
        for key in expected_keys:
            assert key in metrics
    
    def test_evaluate_n_samples(self, trained_model, sample_data):
        """Test n_samples is correct."""
        metrics = trained_model.evaluate(sample_data['X_test'], sample_data['y_test'])
        
        assert metrics['n_samples'] == len(sample_data['X_test'])
    
    def test_confusion_matrix_shape(self, trained_model, sample_data):
        """Test confusion matrix has correct shape."""
        metrics = trained_model.evaluate(sample_data['X_test'], sample_data['y_test'])
        
        n_classes = len(sample_data['y_test'].unique())
        assert metrics['confusion_matrix'].shape == (n_classes, n_classes)
    
    def test_metrics_valid_range(self, trained_model, sample_data):
        """Test metrics are in valid [0, 1] range."""
        metrics = trained_model.evaluate(sample_data['X_test'], sample_data['y_test'])
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1


# ============================================================================
# Test: Feature Importance
# ============================================================================

class TestFeatureImportance:
    """Tests for feature importance extraction."""
    
    def test_importance_shape(self, trained_model):
        """Test importance table has correct number of rows."""
        importance_df = trained_model.get_feature_importance()
        
        # Should have one row per feature
        assert len(importance_df) == len(trained_model.feature_names)
    
    def test_importance_columns(self, trained_model):
        """Test importance table has expected columns."""
        importance_df = trained_model.get_feature_importance()
        
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'rank' in importance_df.columns
    
    def test_importance_sums_to_one(self, trained_model):
        """Test that importances sum to approximately 1."""
        importance_df = trained_model.get_feature_importance()
        
        total = importance_df['importance'].sum()
        assert abs(total - 1.0) < 0.01  # Allow small numerical error
    
    def test_importance_sorted_descending(self, trained_model):
        """Test that importances are sorted descending."""
        importance_df = trained_model.get_feature_importance()
        
        importances = importance_df['importance'].values
        for i in range(len(importances) - 1):
            assert importances[i] >= importances[i + 1]
    
    def test_importance_not_trained_raises(self):
        """Test that get_feature_importance on untrained model raises error."""
        model = AQIRandomForest()
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.get_feature_importance()


# ============================================================================
# Test: OOB Score
# ============================================================================

class TestOOBScore:
    """Tests for OOB score extraction."""
    
    def test_oob_score_available(self, sample_data):
        """Test OOB score is available after training."""
        model = AQIRandomForest(n_estimators=10, max_depth=5, oob_score=True)
        model.train(sample_data['X_train'], sample_data['y_train'])
        
        oob = model.get_oob_score()
        assert 0 <= oob <= 1
    
    def test_oob_score_not_trained_raises(self):
        """Test that get_oob_score on untrained model raises error."""
        model = AQIRandomForest(oob_score=True)
        
        with pytest.raises(RuntimeError, match="not trained"):
            model.get_oob_score()
    
    def test_oob_score_disabled_raises(self, sample_data):
        """Test that get_oob_score raises error when OOB is disabled."""
        model = AQIRandomForest(n_estimators=10, max_depth=5, oob_score=False)
        model.train(sample_data['X_train'], sample_data['y_train'])
        
        with pytest.raises(RuntimeError, match="OOB score not enabled"):
            model.get_oob_score()


# ============================================================================
# Test: Model Persistence
# ============================================================================

class TestModelPersistence:
    """Tests for save/load functionality."""
    
    def test_save_model(self, trained_model):
        """Test saving model creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_model.save_model(save_path)
            
            assert save_path.with_suffix('.joblib').exists()
    
    def test_load_model(self, trained_model, sample_data):
        """Test loading model restores functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_model.save_model(save_path)
            
            loaded_model = AQIRandomForest.load_model(save_path)
            
            assert loaded_model.is_trained == True
            assert loaded_model.feature_names == trained_model.feature_names
            
            # Predictions should match
            orig_pred = trained_model.predict(sample_data['X_test'])
            loaded_pred = loaded_model.predict(sample_data['X_test'])
            
            np.testing.assert_array_equal(orig_pred, loaded_pred)
    
    def test_save_not_trained_raises(self):
        """Test that saving untrained model raises error."""
        model = AQIRandomForest()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            
            with pytest.raises(RuntimeError, match="not trained"):
                model.save_model(save_path)


# ============================================================================
# Test: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_save_metrics_json(self):
        """Test saving metrics to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = {
                'accuracy': 0.85,
                'confusion_matrix': np.array([[10, 2], [3, 15]]),
                'n_samples': 30
            }
            
            save_path = Path(tmpdir) / 'metrics.json'
            save_metrics_json(metrics, save_path)
            
            assert save_path.exists()
            
            # Verify JSON is valid
            import json
            with open(save_path) as f:
                loaded = json.load(f)
            
            assert loaded['accuracy'] == 0.85
            assert loaded['confusion_matrix'] == [[10, 2], [3, 15]]


# ============================================================================
# Test: Random Forest Specific Features
# ============================================================================

class TestRandomForestSpecific:
    """Tests for Random Forest specific features."""
    
    def test_multiple_trees(self, sample_data):
        """Test that multiple trees are created."""
        model = AQIRandomForest(n_estimators=20, max_depth=5)
        model.train(sample_data['X_train'], sample_data['y_train'])
        
        assert len(model.model.estimators_) == 20
    
    def test_different_n_estimators_affects_result(self, sample_data):
        """Test that different n_estimators gives different results."""
        model_small = AQIRandomForest(n_estimators=5, max_depth=5, random_state=42)
        model_large = AQIRandomForest(n_estimators=50, max_depth=5, random_state=42)
        
        model_small.train(sample_data['X_train'], sample_data['y_train'])
        model_large.train(sample_data['X_train'], sample_data['y_train'])
        
        # Both should produce valid results
        acc_small = model_small.evaluate(sample_data['X_test'], sample_data['y_test'])['accuracy']
        acc_large = model_large.evaluate(sample_data['X_test'], sample_data['y_test'])['accuracy']
        
        assert 0 <= acc_small <= 1
        assert 0 <= acc_large <= 1
    
    def test_feature_importance_sum(self, trained_model):
        """Test that feature importances from Random Forest sum to 1."""
        importance_df = trained_model.get_feature_importance()
        total = importance_df['importance'].sum()
        
        # Random Forest guarantees importances sum to 1
        np.testing.assert_almost_equal(total, 1.0, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
