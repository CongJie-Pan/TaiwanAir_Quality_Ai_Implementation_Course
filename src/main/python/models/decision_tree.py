"""
Decision Tree Classification Model for AQI Level Prediction

FR-003: 決策樹分類模型開發

用途：
- 自動學習 AQI 等級分類規則
- 產生可視化決策樹圖
- 對應期中報告的交叉表發現

模型目標：
- 輸入特徵 → 預測 aqi_level (良好/普通/對敏感族群不健康)

課程來源：AIp10 決策與分類
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import json
import joblib

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import sys
# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class AQIDecisionTree:
    """
    Decision Tree classifier for AQI level prediction.
    
    使用 sklearn DecisionTreeClassifier 建立分類模型，
    輸出混淆矩陣、特徵重要性與決策樹視覺化。
    
    Attributes:
        model: sklearn DecisionTreeClassifier instance
        feature_names: List of feature column names
        class_names: List of class labels
        is_trained: Whether the model has been trained
        training_metrics: Metrics from training
    
    Example:
        >>> model = AQIDecisionTree(max_depth=5, class_weight='balanced')
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
        >>> importance_df = model.get_feature_importance()
    """
    
    # AQI 等級標籤 (順序對應分類輸出)
    DEFAULT_CLASS_NAMES = ['良好', '普通', '對敏感族群不健康']
    
    def __init__(
        self, 
        max_depth: int = 5,
        class_weight: str = 'balanced',
        criterion: str = 'gini',
        random_state: int = 42
    ):
        """
        Initialize the Decision Tree classifier.
        
        Args:
            max_depth: Maximum depth of tree (default: 5 for visualization)
            class_weight: 'balanced' to handle class imbalance (per FR-001-D)
            criterion: Split criterion ('gini' or 'entropy')
            random_state: Random seed for reproducibility
        """
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            criterion=criterion,
            random_state=random_state
        )
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.feature_names: List[str] = []
        self.class_names: List[str] = self.DEFAULT_CLASS_NAMES
        self.is_trained = False
        self.training_metrics: Dict = {}
        
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict:
        """
        Train the Decision Tree classifier.
        
        Args:
            X_train: Training features (DataFrame)
            y_train: Training target (Series with aqi_level labels)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dict with training metrics:
            - train_accuracy, train_f1 (macro)
            - val_accuracy, val_f1 (if validation data provided)
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Extract unique class names from training data
        unique_classes = sorted(y_train.unique())
        if len(unique_classes) <= len(self.DEFAULT_CLASS_NAMES):
            self.class_names = unique_classes
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_f1_macro': f1_score(y_train, train_pred, average='macro'),
            'train_samples': len(y_train),
            'tree_depth': self.model.get_depth(),
            'tree_leaves': self.model.get_n_leaves()
        }
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['val_accuracy'] = val_metrics['accuracy']
            metrics['val_f1_macro'] = val_metrics['f1_macro']
            metrics['val_samples'] = len(y_val)
        
        self.training_metrics = metrics
        
        # Print summary
        print(f"\n{'='*50}")
        print("Decision Tree Training Complete")
        print(f"{'='*50}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {metrics['train_samples']:,}")
        print(f"Tree depth: {metrics['tree_depth']} (max={self.max_depth})")
        print(f"Tree leaves: {metrics['tree_leaves']}")
        print(f"\nTraining Metrics:")
        print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
        print(f"  F1 (macro): {metrics['train_f1_macro']:.4f}")
        
        if 'val_accuracy' in metrics:
            print(f"\nValidation Metrics:")
            print(f"  Accuracy:  {metrics['val_accuracy']:.4f}")
            print(f"  F1 (macro): {metrics['val_f1_macro']:.4f}")
        
        print(f"{'='*50}\n")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict AQI level classes.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predicted class labels as numpy array
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Class probabilities as numpy array (n_samples, n_classes)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features DataFrame
            y: True target values
            
        Returns:
            Dict with accuracy, precision, recall, F1, confusion_matrix
        """
        predictions = self.predict(X)
        
        # Get unique labels for metrics calculation
        labels = sorted(y.unique())
        
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision_macro': precision_score(y, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(y, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(y, predictions, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y, predictions, labels=labels),
            'classification_report': classification_report(y, predictions, zero_division=0),
            'n_samples': len(y)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance table.
        
        Returns:
            DataFrame with columns:
            - feature: Feature name
            - importance: Importance value (sum to 1.0)
            - rank: Importance ranking
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Sort by importance
        df = df.sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def get_tree_rules(self) -> str:
        """
        Get text representation of tree decision rules.
        
        Returns:
            String with IF-THEN-ELSE rules
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return export_text(
            self.model, 
            feature_names=self.feature_names,
            class_names=self.class_names if isinstance(self.class_names[0], str) else None
        )
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model (without extension)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'max_depth': self.max_depth,
            'class_weight': self.class_weight,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, path.with_suffix('.joblib'))
        print(f"✅ Model saved to: {path.with_suffix('.joblib')}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'AQIDecisionTree':
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded AQIDecisionTree instance
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.joblib')
        
        model_data = joblib.load(path)
        
        instance = cls(
            max_depth=model_data['max_depth'],
            class_weight=model_data['class_weight']
        )
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.class_names = model_data['class_names']
        instance.training_metrics = model_data['training_metrics']
        instance.is_trained = True
        
        print(f"✅ Model loaded from: {path}")
        return instance


# ============================================================================
# Utility Functions
# ============================================================================

def save_metrics_json(metrics: Dict, path: Union[str, Path]) -> None:
    """Save metrics to JSON file (handle numpy types)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    metrics_converted = convert_numpy(metrics)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics_converted, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metrics saved to: {path}")


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing AQIDecisionTree with synthetic data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Synthetic features
    X = pd.DataFrame({
        'pm2.5_lag1': np.random.uniform(10, 100, n_samples),
        'windspeed': np.random.uniform(0, 10, n_samples),
        'hour_sin': np.random.uniform(-1, 1, n_samples)
    })
    
    # Synthetic target (3 classes based on PM2.5)
    def classify(pm25):
        if pm25 <= 35:
            return '良好'
        elif pm25 <= 70:
            return '普通'
        else:
            return '對敏感族群不健康'
    
    y = pd.Series([classify(p) for p in X['pm2.5_lag1']])
    
    # Train/test split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = AQIDecisionTree(max_depth=5, class_weight='balanced')
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Metrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    print(f"\nFeature Importance:\n{importance_df}")
    
    # Get tree rules (first 500 chars only)
    rules = model.get_tree_rules()
    print(f"\nTree Rules (preview):\n{rules[:500]}")
    
    print("\n✅ AQIDecisionTree test passed!")
