"""
Run Linear Regression Model Training

FR-002: 線性回歸模型正式執行腳本

執行方式:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python src/main/python/models/run_linear_regression.py

產出 (每次執行存到 results/linear_regression/YYYYMMDD_HHMMSS/):
    - coefficients.csv - 係數表
    - metrics.json - R², RMSE, MAE
    - scatter_plot.png - 預測 vs 實際散點圖
    - coefficients_bar.png - 係數條形圖
    - residuals_hist.png - 殘差分佈圖
    - residuals_scatter.png - 殘差 vs 預測值圖
    - model.joblib - 保存的模型
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for saving plots
matplotlib.use('Agg')

# Fix Chinese font display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

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

# Base output directory
RESULTS_BASE = PROJECT_ROOT / 'results' / 'linear_regression'


# ============================================================================
# Chart Functions
# ============================================================================

def create_coefficients_bar_chart(coef_df: pd.DataFrame, save_path: Path) -> None:
    """Create horizontal bar chart of coefficients."""
    # Exclude intercept for visualization
    df = coef_df[coef_df['feature'] != '(intercept)'].copy()
    
    # Sort by coefficient value
    df = df.sort_values('coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color based on positive/negative
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in df['coefficient']]
    
    bars = ax.barh(df['feature'], df['coefficient'], color=colors)
    
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Standardized Coefficient (β)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Linear Regression: Feature Coefficients\n(Red = Positive, Blue = Negative)', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, df['coefficient']):
        x_pos = val + 0.1 if val >= 0 else val - 0.1
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', ha=ha, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Coefficients bar chart: {save_path.name}")


def create_residuals_histogram(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Create histogram of residuals."""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(x=residuals.mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Mean = {residuals.mean():.2f}')
    
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Residual Distribution\n(Std = {residuals.std():.2f})', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Residuals histogram: {save_path.name}")


def create_residuals_scatter(y_pred: np.ndarray, y_true: np.ndarray, save_path: Path) -> None:
    """Create scatter plot of residuals vs predicted values."""
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.3, s=5, c='#3498db')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Predicted AQI', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Residuals vs Predicted Values\n(Check for Heteroscedasticity)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Residuals scatter plot: {save_path.name}")


def create_feature_importance_pie(coef_df: pd.DataFrame, save_path: Path, top_n: int = 8) -> None:
    """Create pie chart of feature importance (absolute coefficients)."""
    # Exclude intercept
    df = coef_df[coef_df['feature'] != '(intercept)'].copy()
    
    # Get top N and group others
    df = df.sort_values('abs_coefficient', ascending=False)
    top_features = df.head(top_n).copy()
    other_sum = df.iloc[top_n:]['abs_coefficient'].sum()
    
    if other_sum > 0:
        other_row = pd.DataFrame({
            'feature': ['Others'],
            'abs_coefficient': [other_sum]
        })
        top_features = pd.concat([top_features[['feature', 'abs_coefficient']], other_row], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_features)))
    
    wedges, texts, autotexts = ax.pie(
        top_features['abs_coefficient'], 
        labels=top_features['feature'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    ax.set_title('Feature Importance (by Absolute Coefficient)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance pie: {save_path.name}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the full Linear Regression training pipeline."""
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_BASE / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FR-002: Linear Regression Model Training")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/6] Loading pre-processed training splits...")
    
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
    
    print(f"  Using {len(available_features)} features")
    
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
    print("\n[2/6] Training Linear Regression model...")
    
    # Train with standardization for coefficient comparison
    model = AQILinearRegression(standardize=True)
    training_metrics = model.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # 3. Evaluate on Test Set
    # =========================================================================
    print("\n[3/6] Evaluating on test set...")
    
    test_metrics = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  R² Score:  {test_metrics['r2']:.4f}")
    print(f"  RMSE:      {test_metrics['rmse']:.2f}")
    print(f"  MAE:       {test_metrics['mae']:.2f}")
    
    # Combine all metrics
    all_metrics = {
        'timestamp': timestamp,
        'features_used': available_features,
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
    print("\n[4/6] Extracting coefficients...")
    
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
    # 5. Generate Charts
    # =========================================================================
    print("\n[5/6] Generating charts...")
    
    # 1. Scatter plot (Predicted vs Actual)
    create_scatter_plot(
        y_test.values, 
        predictions, 
        title=f"Linear Regression: Predicted vs Actual AQI (R²={test_metrics['r2']:.4f})",
        save_path=output_dir / 'scatter_plot.png'
    )
    
    # 2. Coefficients bar chart
    create_coefficients_bar_chart(coef_df, output_dir / 'coefficients_bar.png')
    
    # 3. Residuals histogram
    create_residuals_histogram(y_test.values, predictions, output_dir / 'residuals_hist.png')
    
    # 4. Residuals vs Predicted scatter
    create_residuals_scatter(predictions, y_test.values, output_dir / 'residuals_scatter.png')
    
    # 5. Feature importance pie chart
    create_feature_importance_pie(coef_df, output_dir / 'feature_importance_pie.png')
    
    # =========================================================================
    # 6. Save Results
    # =========================================================================
    print("\n[6/6] Saving results...")
    
    # Save coefficients
    coef_path = output_dir / 'coefficients.csv'
    coef_df.to_csv(coef_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Coefficients: {coef_path.name}")
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    save_metrics_json(all_metrics, metrics_path)
    
    # Save model
    model_path = output_dir / 'model'
    model.save_model(model_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ FR-002 Linear Regression Training Complete!")
    print("=" * 60)
    
    print(f"\n📁 All outputs saved to:\n   {output_dir}")
    print(f"\n📄 Files generated:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")
    
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
    
    return output_dir


if __name__ == "__main__":
    main()
