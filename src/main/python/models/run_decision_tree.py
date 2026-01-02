"""
Run Decision Tree Classification Model Training

FR-003: 決策樹分類模型正式執行腳本

執行方式:
    cd d:/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality
    python src/main/python/models/run_decision_tree.py

產出 (每次執行存到 results/decision_tree/YYYYMMDD_HHMMSS/):
    - confusion_matrix.png - 混淆矩陣熱力圖
    - decision_tree.png - 決策樹視覺化圖
    - feature_importance.png - 特徵重要性條形圖
    - feature_importance.csv - 特徵重要性表格
    - metrics.json - Accuracy, Precision, Recall, F1
    - tree_rules.txt - 文字版決策規則
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

from sklearn.tree import plot_tree

from src.main.python.models.decision_tree import (
    AQIDecisionTree, 
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
MAX_DEPTH = 5  # 限制深度便於可視化
CLASS_WEIGHT = 'balanced'  # 處理類別不平衡 (FR-001-D)

# Base output directory
RESULTS_BASE = PROJECT_ROOT / 'results' / 'decision_tree'


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
        cmap='Blues',
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


def create_decision_tree_plot(
    model: AQIDecisionTree, 
    save_path: Path,
    figsize: tuple = (24, 16)
) -> None:
    """Create visualization of decision tree structure."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot tree
    plot_tree(
        model.model,
        feature_names=model.feature_names,
        class_names=model.class_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    
    ax.set_title(f'Decision Tree (max_depth={model.max_depth})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Decision tree visualization: {save_path.name}")


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
    
    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    
    bars = ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Decision Tree: Feature Importance Ranking', fontsize=14)
    
    # Add value labels
    for bar, val in zip(bars, df['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Feature importance chart: {save_path.name}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the full Decision Tree training pipeline."""
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_BASE / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FR-003: Decision Tree Classification Model Training")
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
    print("\n[2/7] Training Decision Tree classifier...")
    print(f"  Hyperparameters: max_depth={MAX_DEPTH}, class_weight='{CLASS_WEIGHT}'")
    
    model = AQIDecisionTree(
        max_depth=MAX_DEPTH, 
        class_weight=CLASS_WEIGHT
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
    # 5. Get Tree Rules
    # =========================================================================
    print("\n[5/7] Extracting decision tree rules...")
    
    tree_rules = model.get_tree_rules()
    print(f"\n📋 Decision Rules (preview):")
    # Show first 800 chars
    print(tree_rules[:800])
    if len(tree_rules) > 800:
        print("...")
    
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
        title=f"Decision Tree Confusion Matrix (Accuracy={test_metrics['accuracy']:.4f})"
    )
    
    # 2. Decision Tree Visualization
    create_decision_tree_plot(
        model,
        output_dir / 'decision_tree.png'
    )
    
    # 3. Feature Importance Chart
    create_feature_importance_chart(
        importance_df,
        output_dir / 'feature_importance.png'
    )
    
    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("\n[7/7] Saving results...")
    
    # Save feature importance
    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
    print(f"  ✅ Feature importance: {importance_path.name}")
    
    # Save tree rules
    rules_path = output_dir / 'tree_rules.txt'
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write(tree_rules)
    print(f"  ✅ Tree rules: {rules_path.name}")
    
    # Combine all metrics
    all_metrics = {
        'timestamp': timestamp,
        'features_used': available_features,
        'hyperparameters': {
            'max_depth': MAX_DEPTH,
            'class_weight': CLASS_WEIGHT,
            'criterion': 'gini'
        },
        'training': {
            'accuracy': training_metrics['train_accuracy'],
            'f1_macro': training_metrics['train_f1_macro'],
            'samples': training_metrics['train_samples'],
            'tree_depth': training_metrics['tree_depth'],
            'tree_leaves': training_metrics['tree_leaves']
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
    
    # Save model
    model_path = output_dir / 'model'
    model.save_model(model_path)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ FR-003 Decision Tree Training Complete!")
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
    
    # Accuracy interpretation
    accuracy = test_metrics['accuracy']
    if accuracy >= 0.85:
        print(f"\n  💡 Interpretation: Excellent classification (Accuracy ≥ 85%)")
    elif accuracy >= 0.75:
        print(f"\n  💡 Interpretation: Good classification (75% ≤ Accuracy < 85%) ✅ Meets FR-003 target")
    else:
        print(f"\n  💡 Interpretation: Model needs improvement (Accuracy < 75%)")
    
    return output_dir


if __name__ == "__main__":
    main()
