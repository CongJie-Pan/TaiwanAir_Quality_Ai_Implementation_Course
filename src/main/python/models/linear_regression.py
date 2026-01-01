"""
Linear Regression Model for AQI Prediction

FR-002: 線性回歸模型開發

用途：
- 量化風速、季節等因素對 AQI 的影響
- 驗證期中報告「AQI 隨風速增加而下降」的發現

模型公式：
AQI = β₀ + β₁(pm2.5_lag1) + β₂(pm10_lag1) + β₃(o3_lag1) + β₄(windspeed) + ... + ε

課程來源：AIp09 時序與回歸
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import json
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import sys
# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class AQILinearRegression:
    """
    Linear Regression model for AQI prediction.
    
    使用 sklearn LinearRegression 建立多元線性回歸模型，
    輸出係數表與評估指標。
    
    Attributes:
        model: sklearn LinearRegression instance
        scaler: StandardScaler for feature standardization (optional)
        feature_names: List of feature column names
        is_trained: Whether the model has been trained
        standardize: Whether to standardize features
        training_metrics: Metrics from training
    
    Example:
        >>> model = AQILinearRegression(standardize=True)
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
        >>> coef_table = model.get_coefficients()
    """
    
    def __init__(self, standardize: bool = False):
        """
        Initialize the Linear Regression model.
        
        Args:
            standardize: If True, standardize features before training.
                        This makes coefficients directly comparable for
                        feature importance analysis.
        """
        self.model = LinearRegression()
        self.scaler: Optional[StandardScaler] = None
        self.standardize = standardize
        self.feature_names: List[str] = []
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
        Train the Linear Regression model.
        
        Args:
            X_train: Training features (DataFrame)
            y_train: Training target (Series)
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dict with training metrics:
            - train_r2, train_rmse, train_mae
            - val_r2, val_rmse, val_mae (if validation data provided)
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Prepare data
        X_train_processed = X_train.copy()
        
        # Standardize if requested
        if self.standardize:
            self.scaler = StandardScaler()
            X_train_processed = pd.DataFrame(
                self.scaler.fit_transform(X_train_processed),
                columns=self.feature_names,
                index=X_train.index
            )
            print(f"✅ Features standardized (mean≈0, std≈1)")
        
        # Train model
        self.model.fit(X_train_processed, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_processed)
        metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_samples': len(y_train)
        }
        
        # Calculate validation metrics if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            metrics['val_r2'] = val_metrics['r2']
            metrics['val_rmse'] = val_metrics['rmse']
            metrics['val_mae'] = val_metrics['mae']
            metrics['val_samples'] = len(y_val)
        
        self.training_metrics = metrics
        
        # Print summary
        print(f"\n{'='*50}")
        print("Linear Regression Training Complete")
        print(f"{'='*50}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Training samples: {metrics['train_samples']:,}")
        print(f"\nTraining Metrics:")
        print(f"  R² Score:  {metrics['train_r2']:.4f}")
        print(f"  RMSE:      {metrics['train_rmse']:.2f}")
        print(f"  MAE:       {metrics['train_mae']:.2f}")
        
        if 'val_r2' in metrics:
            print(f"\nValidation Metrics:")
            print(f"  R² Score:  {metrics['val_r2']:.4f}")
            print(f"  RMSE:      {metrics['val_rmse']:.2f}")
            print(f"  MAE:       {metrics['val_mae']:.2f}")
        
        print(f"{'='*50}\n")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict AQI values.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predicted AQI values as numpy array
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        X_processed = X.copy()
        
        # Apply same standardization
        if self.standardize and self.scaler is not None:
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X.columns,
                index=X.index
            )
        
        return self.model.predict(X_processed)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features DataFrame
            y: True target values
            
        Returns:
            Dict with R², RMSE, MAE metrics
        """
        predictions = self.predict(X)
        
        return {
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'n_samples': len(y)
        }
    
    def get_coefficients(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get coefficient table with interpretation.
        
        Args:
            feature_names: Override feature names (optional)
            
        Returns:
            DataFrame with columns:
            - feature: Feature name
            - coefficient: β value
            - abs_coefficient: Absolute value for ranking
            - interpretation: Human-readable interpretation
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        names = feature_names or self.feature_names
        coefs = self.model.coef_
        intercept = self.model.intercept_
        
        # Build coefficient table
        data = []
        for name, coef in zip(names, coefs):
            # Generate interpretation
            if coef > 0:
                interpretation = f"每增加1單位，AQI 增加 {abs(coef):.2f} 點"
            else:
                interpretation = f"每增加1單位，AQI 減少 {abs(coef):.2f} 點"
            
            data.append({
                'feature': name,
                'coefficient': coef,
                'abs_coefficient': abs(coef),
                'interpretation': interpretation
            })
        
        # Add intercept
        data.append({
            'feature': '(intercept)',
            'coefficient': intercept,
            'abs_coefficient': abs(intercept),
            'interpretation': f'基準 AQI 值 (所有特徵為0時)'
        })
        
        df = pd.DataFrame(data)
        
        # Sort by absolute coefficient (excluding intercept)
        df_features = df[df['feature'] != '(intercept)'].sort_values(
            'abs_coefficient', ascending=False
        )
        df_intercept = df[df['feature'] == '(intercept)']
        
        return pd.concat([df_features, df_intercept], ignore_index=True)
    
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
            'scaler': self.scaler,
            'standardize': self.standardize,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, path.with_suffix('.joblib'))
        print(f"✅ Model saved to: {path.with_suffix('.joblib')}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'AQILinearRegression':
        """
        Load model from disk.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded AQILinearRegression instance
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.joblib')
        
        model_data = joblib.load(path)
        
        instance = cls(standardize=model_data['standardize'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.is_trained = True
        
        print(f"✅ Model loaded from: {path}")
        return instance


# ============================================================================
# Utility Functions
# ============================================================================

def create_scatter_plot(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    title: str = "Predicted vs Actual AQI",
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Create scatter plot of predicted vs actual values.
    
    Args:
        y_true: True AQI values
        y_pred: Predicted AQI values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels
    ax.set_xlabel('Actual AQI', fontsize=12)
    ax.set_ylabel('Predicted AQI', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Calculate and show R²
    r2 = r2_score(y_true, y_pred)
    ax.text(
        0.05, 0.95, f'R² = {r2:.4f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Scatter plot saved to: {save_path}")
    
    plt.close()


def save_metrics_json(metrics: Dict, path: Union[str, Path]) -> None:
    """Save metrics to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Metrics saved to: {path}")


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing AQILinearRegression with synthetic data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Synthetic features
    X = pd.DataFrame({
        'pm2.5_lag1': np.random.uniform(10, 100, n_samples),
        'windspeed': np.random.uniform(0, 10, n_samples),
        'hour': np.random.randint(0, 24, n_samples)
    })
    
    # Synthetic target (AQI roughly correlates with PM2.5, inversely with windspeed)
    y = pd.Series(
        30 + 0.5 * X['pm2.5_lag1'] - 3 * X['windspeed'] + np.random.normal(0, 10, n_samples)
    )
    
    # Train/test split
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train model
    model = AQILinearRegression(standardize=True)
    model.train(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Metrics: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.2f}")
    
    # Get coefficients
    coef_df = model.get_coefficients()
    print(f"\nCoefficients:\n{coef_df}")
    
    print("\n✅ AQILinearRegression test passed!")
