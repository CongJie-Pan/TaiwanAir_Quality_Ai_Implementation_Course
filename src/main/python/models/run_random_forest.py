"""
Run Random Forest Classification Model Training

FR-004: 隨機森林分類模型正式執行腳本

執行方式:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python src/main/python/models/run_random_forest.py

產出 (每次執行存到 results/random_forest/YYYYMMDD_HHMMSS/):
    - confusion_matrix.png - 混淆矩陣熱力圖
    - feature_importance.png - 特徵重要性條形圖
    - feature_importance.csv - 特徵重要性表格
    - metrics.json - Accuracy, Precision, Recall, F1, OOB Score
    - model_comparison.md - 與決策樹的準確度比較表
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
import seaborn as sns

# Use non-interactive backend for saving plots
matplotlib.use('Agg')

# Fix Chinese font display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from src.main.python.models.random_forest import (
    AQIRandomForest, 
    save_metrics_json
)
from src.main.python.models.data_preprocessing import load_training_splits


# ============================================================================
# Configuration
# ============================================================================

# Feature columns to use (for tree models, Label Encoding is acceptable)
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
    # County - use Label Encoding for tree models (more efficient)
    'county_encoded',
]

# Target column
TARGET_COL = 'aqi_level'

# Model hyperparameters (from 02-models.md spec)
N_ESTIMATORS = 100  # 森林中決策樹數量
MAX_DEPTH = 10  # 每棵樹最大深度
CLASS_WEIGHT = 'balanced'  # 處理類別不平衡 (FR-001-D)

# Base output directory
RESULTS_BASE = PROJECT_ROOT / 'results' / 'random_forest'

# Decision tree results for comparison (find latest)
DECISION_TREE_RESULTS = PROJECT_ROOT / 'results' / 'decision_tree'


# ============================================================================
# Chart Functions
# ============================================================================

def create_confusion_matrix_plot(
    confusion_mat: np.ndarray, 
    class_names: list, 
    save_path: Path,
    title: str = "Confusion Matrix"
) -> None:
    """Create heatmap visualization of confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        confusion_mat, 
        annot=True, 
        fmt='d', 
        cmap='Greens',  # Use green to differentiate from Decision Tree (blue)
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix: {save_path.name}")


def create_feature_importance_chart(
    importance_df: pd.DataFrame, 
    save_path: Path,
    top_n: int = 15
) -> None:
    """Create horizontal bar chart of feature importance."""
    # Get top N features
    df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort for horizontal bar (smallest at top)
    df = df.sort_values('importance', ascending=True)
    
    # Color gradient (green for Random Forest)
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(df)))
    
    bars = ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Random Forest: Feature Importance Ranking', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, df['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance chart: {save_path.name}")


def create_model_comparison_chart(
    rf_accuracy: float,
    dt_accuracy: float,
    rf_f1: float,
    dt_f1: float,
    save_path: Path
) -> None:
    """Create bar chart comparing Random Forest vs Decision Tree."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'F1 (Macro)']
    rf_scores = [rf_accuracy, rf_f1]
    dt_scores = [dt_accuracy, dt_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='#3498db')
    bars2 = ax.bar(x + width/2, rf_scores, width, label='Random Forest', color='#27ae60')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: Decision Tree vs Random Forest', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Model comparison chart: {save_path.name}")


def get_latest_decision_tree_metrics() -> dict:
    """Find and load the latest Decision Tree metrics for comparison."""
    if not DECISION_TREE_RESULTS.exists():
        return None
    
    # Find latest timestamp folder
    folders = [f for f in DECISION_TREE_RESULTS.iterdir() if f.is_dir()]
    if not folders:
        return None
    
    latest_folder = max(folders, key=lambda x: x.name)
    metrics_path = latest_folder / 'metrics.json'
    
    if not metrics_path.exists():
        return None
    
    import json
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_model_comparison_markdown(
    rf_metrics: dict,
    dt_metrics: dict,
    save_path: Path
) -> None:
    """Create markdown file comparing models."""
    content = f"""# Model Comparison: Decision Tree vs Random Forest

## Test Set Performance

| Metric | Decision Tree | Random Forest | Improvement |
|--------|---------------|---------------|-------------|
| **Accuracy** | {dt_metrics['test']['accuracy']:.4f} | {rf_metrics['test']['accuracy']:.4f} | {(rf_metrics['test']['accuracy'] - dt_metrics['test']['accuracy'])*100:+.2f}% |
| **Precision (Macro)** | {dt_metrics['test']['precision_macro']:.4f} | {rf_metrics['test']['precision_macro']:.4f} | {(rf_metrics['test']['precision_macro'] - dt_metrics['test']['precision_macro'])*100:+.2f}% |
| **Recall (Macro)** | {dt_metrics['test']['recall_macro']:.4f} | {rf_metrics['test']['recall_macro']:.4f} | {(rf_metrics['test']['recall_macro'] - dt_metrics['test']['recall_macro'])*100:+.2f}% |
| **F1 (Macro)** | {dt_metrics['test']['f1_macro']:.4f} | {rf_metrics['test']['f1_macro']:.4f} | {(rf_metrics['test']['f1_macro'] - dt_metrics['test']['f1_macro'])*100:+.2f}% |

## Model Configuration

| Parameter | Decision Tree | Random Forest |
|-----------|---------------|---------------|
| **max_depth** | {dt_metrics['hyperparameters']['max_depth']} | {rf_metrics['hyperparameters']['max_depth']} |
| **n_estimators** | 1 | {rf_metrics['hyperparameters']['n_estimators']} |
| **class_weight** | {dt_metrics['hyperparameters']['class_weight']} | {rf_metrics['hyperparameters']['class_weight']} |

## OOB Score (Random Forest Only)

Random Forest OOB Score: **{rf_metrics['training']['oob_score']:.4f}**

> OOB (Out-of-Bag) Score 是隨機森林特有的交叉驗證指標，利用未被抽樣的資料進行驗證，
> 提供模型泛化能力的無偏估計。

## Conclusion

"""
    
    rf_acc = rf_metrics['test']['accuracy']
    dt_acc = dt_metrics['test']['accuracy']
    
    if rf_acc > dt_acc:
        content += f"隨機森林模型 (Accuracy={rf_acc:.4f}) 優於決策樹模型 (Accuracy={dt_acc:.4f})，提升了 {(rf_acc-dt_acc)*100:.2f}%。\n"
    elif rf_acc < dt_acc:
        content += f"決策樹模型 (Accuracy={dt_acc:.4f}) 優於隨機森林模型 (Accuracy={rf_acc:.4f})。\n"
    else:
        content += f"兩個模型效能相同 (Accuracy={rf_acc:.4f})。\n"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✅ Model comparison: {save_path.name}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the full Random Forest training pipeline."""
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_BASE / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FR-004: Random Forest Classification Model Training")
    print("=" * 60)
    print(f"📁 Output directory: {output_dir}")
    
    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n[1/7] Loading pre-processed training splits...")
    
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
    
    # Check target column
    if TARGET_COL not in df_train.columns:
        print(f"❌ Error: Target column '{TARGET_COL}' not found in data")
        return
    
    # Prepare X and y (Classification task - use aqi_level)
    X_train = df_train[available_features]
    y_train = df_train[TARGET_COL]
    
    X_val = df_val[available_features]
    y_val = df_val[TARGET_COL]
    
    X_test = df_test[available_features]
    y_test = df_test[TARGET_COL]
    
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Show class distribution
    print(f"\n  Class distribution (train):")
    for cls, count in y_train.value_counts().items():
        pct = count / len(y_train) * 100
        print(f"    {cls}: {count:,} ({pct:.1f}%)")
    
    # =========================================================================
    # 2. Train Model
    # =========================================================================
    print("\n[2/7] Training Random Forest classifier...")
    print(f"  Hyperparameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, class_weight='{CLASS_WEIGHT}'")
    
    model = AQIRandomForest(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH, 
        class_weight=CLASS_WEIGHT,
        oob_score=True
    )
    training_metrics = model.train(X_train, y_train, X_val, y_val)
    
    # =========================================================================
    # 3. Evaluate on Test Set
    # =========================================================================
    print("\n[3/7] Evaluating on test set...")
    
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"\n📊 Test Set Performance:")
    print(f"  Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"  Precision (M): {test_metrics['precision_macro']:.4f}")
    print(f"  Recall (M):    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):    {test_metrics['f1_macro']:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(test_metrics['classification_report'])
    
    # =========================================================================
    # 4. Extract Feature Importance
    # =========================================================================
    print("\n[4/7] Extracting feature importance...")
    
    importance_df = model.get_feature_importance()
    
    print("\n📋 Top 10 Feature Importance:")
    print(importance_df.head(10).to_string(index=False))
    
    # =========================================================================
    # 5. Load Decision Tree Metrics for Comparison
    # =========================================================================
    print("\n[5/7] Loading Decision Tree metrics for comparison...")
    
    dt_metrics = get_latest_decision_tree_metrics()
    if dt_metrics:
        print(f"  Found Decision Tree metrics from: {dt_metrics['timestamp']}")
        print(f"  DT Test Accuracy: {dt_metrics['test']['accuracy']:.4f}")
    else:
        print("  ⚠️ No Decision Tree metrics found for comparison")
    
    # =========================================================================
    # 6. Generate Charts
    # =========================================================================
    print("\n[6/7] Generating charts...")
    
    # Get class names for confusion matrix
    class_names = sorted(y_test.unique())
    
    # 1. Confusion Matrix
    create_confusion_matrix_plot(
        test_metrics['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix.png',
        title=f"Random Forest Confusion Matrix (Accuracy={test_metrics['accuracy']:.4f})"
    )
    
    # 2. Feature Importance Chart
    create_feature_importance_chart(
        importance_df,
        output_dir / 'feature_importance.png'
    )
    
    # 3. Model Comparison Chart (if DT metrics available)
    if dt_metrics:
        create_model_comparison_chart(
            rf_accuracy=test_metrics['accuracy'],
            dt_accuracy=dt_metrics['test']['accuracy'],
            rf_f1=test_metrics['f1_macro'],
            dt_f1=dt_metrics['test']['f1_macro'],
            save_path=output_dir / 'model_comparison.png'
        )
    
    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("\n[7/7] Saving results...")
    
    # Save feature importance
    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Feature importance: {importance_path.name}")
    
    # Combine all metrics
    all_metrics = {
        'timestamp': timestamp,
        'features_used': available_features,
        'hyperparameters': {
            'n_estimators': N_ESTIMATORS,
            'max_depth': MAX_DEPTH,
            'class_weight': CLASS_WEIGHT
        },
        'training': {
            'accuracy': training_metrics['train_accuracy'],
            'f1_macro': training_metrics['train_f1_macro'],
            'samples': training_metrics['train_samples'],
            'oob_score': training_metrics.get('oob_score')
        },
        'validation': {
            'accuracy': training_metrics.get('val_accuracy'),
            'f1_macro': training_metrics.get('val_f1_macro'),
            'samples': training_metrics.get('val_samples')
        },
        'test': {
            'accuracy': test_metrics['accuracy'],
            'precision_macro': test_metrics['precision_macro'],
            'recall_macro': test_metrics['recall_macro'],
            'f1_macro': test_metrics['f1_macro'],
            'n_samples': test_metrics['n_samples'],
            'confusion_matrix': test_metrics['confusion_matrix']
        }
    }
    
    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    save_metrics_json(all_metrics, metrics_path)
    
    # Save model comparison markdown (if DT metrics available)
    if dt_metrics:
        create_model_comparison_markdown(
            all_metrics,
            dt_metrics,
            output_dir / 'model_comparison.md'
        )
    
    # Save model
    model_path = output_dir / 'model'
    model.save_model(model_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ FR-004 Random Forest Training Complete!")
    print("=" * 60)
    
    print(f"\n📁 All outputs saved to:\n   {output_dir}")
    print(f"\n📄 Files generated:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")
    
    print(f"\n📈 Model Performance Summary:")
    print(f"  Train Accuracy: {training_metrics['train_accuracy']:.4f}")
    if 'val_accuracy' in training_metrics:
        print(f"  Val Accuracy:   {training_metrics['val_accuracy']:.4f}")
    print(f"  Test Accuracy:  {test_metrics['accuracy']:.4f}")
    if 'oob_score' in training_metrics:
        print(f"  OOB Score:      {training_metrics['oob_score']:.4f}")
    
    # Comparison with Decision Tree
    if dt_metrics:
        dt_acc = dt_metrics['test']['accuracy']
        rf_acc = test_metrics['accuracy']
        diff = (rf_acc - dt_acc) * 100
        print(f"\n📊 Comparison with Decision Tree:")
        print(f"  Decision Tree: {dt_acc:.4f}")
        print(f"  Random Forest: {rf_acc:.4f}")
        if diff > 0:
            print(f"  Improvement:   +{diff:.2f}% ✅")
        elif diff < 0:
            print(f"  Difference:    {diff:.2f}%")
        else:
            print(f"  Same performance")
    
    # Accuracy interpretation
    accuracy = test_metrics['accuracy']
    if accuracy >= 0.85:
        print(f"\n  💡 Interpretation: Excellent classification (Accuracy ≥ 85%)")
    elif accuracy >= 0.80:
        print(f"\n  💡 Interpretation: Very good classification (80% ≤ Accuracy < 85%) ✅ Meets FR-004 target")
    elif accuracy >= 0.75:
        print(f"\n  💡 Interpretation: Good classification (75% ≤ Accuracy < 80%)")
    else:
        print(f"\n  💡 Interpretation: Model needs improvement (Accuracy < 75%)")
    
    return output_dir


if __name__ == "__main__":
    main()
