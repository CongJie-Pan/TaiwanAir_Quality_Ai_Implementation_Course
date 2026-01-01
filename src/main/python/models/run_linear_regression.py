"""
Run Linear Regression Model Training

FR-002: 線性回歸模型正式執行腳本

執行方式:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python src/main/python/models/run_linear_regression.py

產出:
    - results/linear_regression_coefficients.csv - 係數表
    - results/linear_regression_metrics.json - R², RMSE, MAE
    - results/linear_regression_scatter.png - 預測 vs 實際散點圖
    - models/saved/linear_regression_model.joblib - 保存的模型
"""

import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.main.python.models.linear_regression import (
    AQILinearRegression, 
    create_scatter_plot, 
    save_metrics_json
)
from src.main.python.models.data_preprocessing import load_training_splits


# ============================================================================
# Configuration
# ============================================================================

# Feature columns to use (based on FR-001 data preparation)
FEATURE_COLS = [
    # Lag features (避免 Data Leakage)
    'pm2.5_lag1', 'pm10_lag1', 'o3_lag1',
    'pm2.5_lag24', 'pm10_lag24', 'o3_lag24',
    # Meteorological & temporal
    'windspeed',
    # Cyclical time features (FR-001-H)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    # Season One-Hot (FR-001-C)
    'season_spring', 'season_summer', 'season_autumn', 'season_winter',
    # County One-Hot (FR-001-I) - for linear regression, use One-Hot to avoid ordinal assumption
    'county_new_taipei', 'county_changhua', 'county_kaohsiung',
]

# Output directories
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved'


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the full Linear Regression training pipeline."""
    
    print("=" * 60)
    print("FR-002: Linear Regression Model Training")
    print("=" * 60)
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/5] Loading pre-processed training splits...")
    
    try:
        df_train, df_val, df_test = load_training_splits()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Please run save_multiyear_splits() first to generate data.")
        return
    
    # Filter to available feature columns
    available_features = [col for col in FEATURE_COLS if col in df_train.columns]
    missing_features = [col for col in FEATURE_COLS if col not in df_train.columns]
    
    if missing_features:
        print(f"  ⚠️ Missing features (will be skipped): {missing_features}")
    
    print(f"  Using {len(available_features)} features: {available_features}")
    
    # Prepare X and y
    X_train = df_train[available_features]
    y_train = df_train['aqi']
    
    X_val = df_val[available_features]
    y_val = df_val['aqi']
    
    X_test = df_test[available_features]
    y_test = df_test['aqi']
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # =========================================================================
    # 2. Train Model
    # =========================================================================
    print("\n[2/5] Training Linear Regression model...")
    
    # Train with standardization for coefficient comparison
    model = AQILinearRegression(standardize=True)
    training_metrics = model.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # 3. Evaluate on Test Set
    # =========================================================================
    print("\n[3/5] Evaluating on test set...")
    
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  R² Score:  {test_metrics['r2']:.4f}")
    print(f"  RMSE:      {test_metrics['rmse']:.2f}")
    print(f"  MAE:       {test_metrics['mae']:.2f}")
    
    # Combine all metrics
    all_metrics = {
        'training': {
            'r2': training_metrics['train_r2'],
            'rmse': training_metrics['train_rmse'],
            'mae': training_metrics['train_mae'],
            'samples': training_metrics['train_samples']
        },
        'validation': {
            'r2': training_metrics.get('val_r2'),
            'rmse': training_metrics.get('val_rmse'),
            'mae': training_metrics.get('val_mae'),
            'samples': training_metrics.get('val_samples')
        },
        'test': test_metrics
    }
    
    # =========================================================================
    # 4. Extract Coefficients
    # =========================================================================
    print("\n[4/5] Extracting coefficients...")
    
    coef_df = model.get_coefficients()
    
    print("\n📋 Top 10 Feature Coefficients (by importance):")
    print(coef_df.head(10).to_string(index=False))
    
    # Validate midterm finding: windspeed should have negative coefficient
    windspeed_coef = coef_df[coef_df['feature'] == 'windspeed']['coefficient'].values
    if len(windspeed_coef) > 0:
        if windspeed_coef[0] < 0:
            print(f"\n✅ Midterm finding validated: windspeed coefficient = {windspeed_coef[0]:.4f} (negative)")
            print("   → Higher wind speed correlates with lower AQI")
        else:
            print(f"\n⚠️ Unexpected: windspeed coefficient = {windspeed_coef[0]:.4f} (positive)")
    
    # =========================================================================
    # 5. Save Results
    # =========================================================================
    print("\n[5/5] Saving results...")
    
    # Create output directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save coefficients
    coef_path = RESULTS_DIR / 'linear_regression_coefficients.csv'
    coef_df.to_csv(coef_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Coefficients: {coef_path}")
    
    # Save metrics
    metrics_path = RESULTS_DIR / 'linear_regression_metrics.json'
    save_metrics_json(all_metrics, metrics_path)
    
    # Save scatter plot
    predictions = model.predict(X_test)
    scatter_path = RESULTS_DIR / 'linear_regression_scatter.png'
    create_scatter_plot(
        y_test.values, 
        predictions, 
        title=f"Linear Regression: Predicted vs Actual AQI (R²={test_metrics['r2']:.4f})",
        save_path=scatter_path
    )
    
    # Save model
    model_path = MODELS_DIR / 'linear_regression_model'
    model.save_model(model_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ FR-002 Linear Regression Training Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  📄 {coef_path}")
    print(f"  📄 {metrics_path}")
    print(f"  📊 {scatter_path}")
    print(f"  🧠 {model_path}.joblib")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"  Train R²: {training_metrics['train_r2']:.4f}")
    if 'val_r2' in training_metrics:
        print(f"  Val R²:   {training_metrics['val_r2']:.4f}")
    print(f"  Test R²:  {test_metrics['r2']:.4f}")
    
    # R² interpretation
    r2 = test_metrics['r2']
    if r2 >= 0.75:
        print(f"\n  💡 Interpretation: Excellent fit (R² ≥ 0.75)")
    elif r2 >= 0.5:
        print(f"\n  💡 Interpretation: Acceptable fit (0.5 ≤ R² < 0.75)")
    else:
        print(f"\n  💡 Interpretation: Weak fit (R² < 0.5) - Linear model may be insufficient")


if __name__ == "__main__":
    main()
