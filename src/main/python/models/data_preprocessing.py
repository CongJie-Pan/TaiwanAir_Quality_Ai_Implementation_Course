"""
Data Preprocessing Module for Air Quality Prediction Models

This module provides feature engineering and data preparation functions
for training machine learning models on air quality data.

Features:
- Season labeling (春/夏/秋/冬)
- AQI level classification (良好/普通/對敏感族群不健康)
- Wind speed level classification (無風/輕風/微風以上)
- Train/validation/test split (70/15/15)

Reference: docs/finalReport/spec/01-problem-and-data.md

Author: AI Assistant
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.main.python.utils.data_loader import AirQualityDataLoader


# ============================================================================
# Feature Engineering Functions
# ============================================================================

def add_season(df: pd.DataFrame, month_col: str = 'month') -> pd.DataFrame:
    """
    Add season label based on month.
    
    季節對應：
    - 冬季: 12, 1, 2月
    - 春季: 3, 4, 5月
    - 夏季: 6, 7, 8月
    - 秋季: 9, 10, 11月
    
    Args:
        df: DataFrame with month column
        month_col: Name of month column
        
    Returns:
        DataFrame with 'season' column added
    """
    season_map = {
        12: '冬季', 1: '冬季', 2: '冬季',
        3: '春季', 4: '春季', 5: '春季',
        6: '夏季', 7: '夏季', 8: '夏季',
        9: '秋季', 10: '秋季', 11: '秋季'
    }
    
    df = df.copy()
    df['season'] = df[month_col].map(season_map)
    return df


def add_aqi_level(df: pd.DataFrame, aqi_col: str = 'aqi') -> pd.DataFrame:
    """
    Add AQI level classification.
    
    AQI 等級標準：
    - 良好: AQI <= 50
    - 普通: 50 < AQI <= 100
    - 對敏感族群不健康: AQI > 100
    
    Args:
        df: DataFrame with AQI column
        aqi_col: Name of AQI column
        
    Returns:
        DataFrame with 'aqi_level' column added
    """
    def classify_aqi(aqi):
        if pd.isna(aqi):
            return None
        if aqi <= 50:
            return '良好'
        elif aqi <= 100:
            return '普通'
        else:
            return '對敏感族群不健康'
    
    df = df.copy()
    df['aqi_level'] = df[aqi_col].apply(classify_aqi)
    return df


def add_wind_level(df: pd.DataFrame, wind_col: str = 'windspeed') -> pd.DataFrame:
    """
    Add wind speed level classification.
    
    風速等級標準（對應期中報告交叉表）：
    - 無風: windspeed <= 1.5 m/s
    - 輕風: 1.5 < windspeed <= 3.4 m/s
    - 微風以上: windspeed > 3.4 m/s
    
    Args:
        df: DataFrame with windspeed column
        wind_col: Name of windspeed column
        
    Returns:
        DataFrame with 'wind_level' column added
    """
    df = df.copy()
    df['wind_level'] = pd.cut(
        df[wind_col],
        bins=[0, 1.5, 3.4, float('inf')],
        labels=['無風', '輕風', '微風以上'],
        include_lowest=True
    )
    return df


# ============================================================================
# Data Saving and Loading (Cleaned Data)
# ============================================================================

CLEANED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'model_used' / 'cleaned'

# 模型訓練需要的欄位 (精簡版)
MODEL_COLS = [
    'date',           # 時間戳（參考用）
    'county',         # 原始縣市名
    'aqi',            # 回歸目標
    'aqi_level',      # 分類目標
    'windspeed',      # 特徵
    'month', 'hour',  # 時間特徵
    'season', 'season_encoded',      # 季節
    'county_encoded',                 # 空間
    'wind_level', 'wind_level_encoded'  # 風速等級
]


def save_cleaned_data(df: pd.DataFrame, year: int = 2023) -> dict:
    """
    Save cleaned data to two parquet files:
    1. cleaned_{year}_full.parquet - 保留所有欄位（供分析用）
    2. cleaned_{year}_model.parquet - 只保留模型需要的欄位（供訓練用）
    
    儲存位置: data/processed/model_used/cleaned/
    
    Args:
        df: Cleaned DataFrame to save
        year: Data year (default: 2023)
        
    Returns:
        Dict with paths to both saved files
    """
    # Create directory if not exists
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Save full version (all columns)
    full_path = CLEANED_DATA_DIR / f'cleaned_{year}_full.parquet'
    df.to_parquet(full_path, index=False, compression='snappy')
    full_size_mb = full_path.stat().st_size / (1024 * 1024)
    results['full'] = full_path
    
    print(f"✅ Saved FULL version: {full_path}")
    print(f"   File size: {full_size_mb:.2f} MB | Columns: {len(df.columns)} | Rows: {len(df):,}")
    
    # 2. Save model version (only essential columns)
    available_cols = [col for col in MODEL_COLS if col in df.columns]
    df_model = df[available_cols]
    
    model_path = CLEANED_DATA_DIR / f'cleaned_{year}_model.parquet'
    df_model.to_parquet(model_path, index=False, compression='snappy')
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    results['model'] = model_path
    
    print(f"✅ Saved MODEL version: {model_path}")
    print(f"   File size: {model_size_mb:.2f} MB | Columns: {len(df_model.columns)} | Rows: {len(df_model):,}")
    
    return results


def load_cleaned_data(filename: str = 'cleaned_2023_model.parquet') -> pd.DataFrame:
    """
    Load previously saved cleaned data.
    
    Args:
        filename: Filename to load (default: model version)
        
    Returns:
        Cleaned DataFrame
        
    Raises:
        FileNotFoundError: If cleaned data file doesn't exist
    """
    file_path = CLEANED_DATA_DIR / filename
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {file_path}\n"
            f"Run save_cleaned_data() first to generate it."
        )
    
    df = pd.read_parquet(file_path)
    print(f"✅ Loaded cleaned data from: {file_path}")
    print(f"   Rows: {len(df):,} | Columns: {len(df.columns)}")
    
    return df


def cleaned_data_exists(year: int = 2023, version: str = 'model') -> bool:
    """Check if cleaned data file exists."""
    filename = f'cleaned_{year}_{version}.parquet'
    return (CLEANED_DATA_DIR / filename).exists()


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_training_data(
    year: int = 2023,
    counties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load training data from parquet files.
    
    Args:
        year: Year of data to load (default: 2023)
        counties: List of counties to filter (default: New Taipei City, Changhua County, Kaohsiung City)
        
    Returns:
        Raw DataFrame loaded from data source
    """
    if counties is None:
        # 延續期中報告分析範圍 (English names in dataset)
        counties = ['New Taipei City', 'Changhua County', 'Kaohsiung City']
    
    loader = AirQualityDataLoader(data_dir=str(PROJECT_ROOT))
    
    # Load data for specified year
    df = loader.load_by_year(year)
    
    # Filter by counties
    if counties:
        df = df[df['county'].isin(counties)]
    
    loader.close()
    
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with all derived features added
    """
    # Ensure month column exists
    if 'month' not in df.columns and 'date' in df.columns:
        df = df.copy()
        df['month'] = pd.to_datetime(df['date']).dt.month
    
    # Ensure hour column exists  
    if 'hour' not in df.columns and 'date' in df.columns:
        df['hour'] = pd.to_datetime(df['date']).dt.hour
    
    # Apply feature engineering
    df = add_season(df)
    df = add_aqi_level(df)
    df = add_wind_level(df)
    
    return df


def clean_data(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Clean data by removing rows with missing values and invalid sentinel values.
    
    處理項目：
    1. 移除 required_cols 中有 NaN 的列
    2. 移除 AQI < 0 的列 (AQI=-1 是哨兵值，代表無效資料)
    
    Args:
        df: DataFrame to clean
        required_cols: List of columns that must not have NaN
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    initial_len = len(df)
    
    # Check which required columns exist
    existing_cols = [col for col in required_cols if col in df.columns]
    
    # Step 1: Drop rows with NaN in required columns
    df = df.dropna(subset=existing_cols)
    dropped_nan = initial_len - len(df)
    
    # Step 2: Remove sentinel values (AQI = -1 means invalid/missing data)
    if 'aqi' in df.columns:
        before_sentinel = len(df)
        df = df[df['aqi'] >= 0]
        dropped_sentinel = before_sentinel - len(df)
        if dropped_sentinel > 0:
            print(f"Removed {dropped_sentinel:,} rows with invalid AQI (sentinel value -1)")
    else:
        dropped_sentinel = 0
    
    total_dropped = initial_len - len(df)
    if total_dropped > 0:
        print(f"Total cleaned: {total_dropped:,} rows ({total_dropped/initial_len*100:.2f}%)")
        print(f"  - NaN values: {dropped_nan:,}")
        print(f"  - Sentinel values (AQI<0): {dropped_sentinel:,}")
    
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    cat_cols: List[str] = ['season', 'county', 'wind_level']
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df: DataFrame with categorical columns
        cat_cols: List of categorical column names
        
    Returns:
        Tuple of (encoded DataFrame, dict of encoders)
    """
    df = df.copy()
    encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Handle NaN by converting to string first
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets (70/15/15).
    
    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for test set (default: 0.15)
        val_size: Proportion for validation set (default: 0.15)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    # Adjust val_size to account for already removed test set
    val_adjusted = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_adjusted,
        random_state=random_state
    )
    
    print(f"Data split complete:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# High-Level Data Preparation Functions
# ============================================================================

def prepare_regression_data(
    year: int = 2023,
    counties: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for regression task (predicting AQI value).
    
    Args:
        year: Year of data to use
        counties: Counties to include
        feature_cols: Feature columns (default: windspeed, month, hour, pm2.5, pm10, o3)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, encoders)
    """
    if feature_cols is None:
        feature_cols = ['windspeed', 'month', 'hour', 'season_encoded', 'county_encoded']
    
    # Load and prepare data
    df = load_training_data(year=year, counties=counties)
    df = apply_feature_engineering(df)
    
    # Clean data
    required = ['aqi', 'windspeed', 'month']
    df = clean_data(df, required)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Select features and target
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    y = df['aqi']
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


def prepare_classification_data(
    year: int = 2023,
    counties: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for classification task (predicting AQI level).
    
    Args:
        year: Year of data to use
        counties: Counties to include
        feature_cols: Feature columns
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, encoders)
    """
    if feature_cols is None:
        feature_cols = ['windspeed', 'month', 'hour', 'season_encoded', 'county_encoded']
    
    # Load and prepare data
    df = load_training_data(year=year, counties=counties)
    df = apply_feature_engineering(df)
    
    # Clean data - need valid AQI to create aqi_level
    required = ['aqi', 'aqi_level', 'windspeed', 'month']
    df = clean_data(df, required)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['aqi_level'])
    encoders['aqi_level'] = le_target
    
    # Select features
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features]
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, pd.Series(y))
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Preprocessing for Air Quality Models')
    parser.add_argument('--save', action='store_true', help='Save cleaned data to parquet file')
    parser.add_argument('--test', action='store_true', help='Run data preparation tests')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Data Preprocessing Module")
    print("=" * 60)
    
    if args.save or (not args.save and not args.test):
        # Generate and save cleaned data
        print("\n[Generating Cleaned Data]")
        
        # Load raw data
        df = load_training_data(year=2023)
        print(f"Raw data loaded: {len(df):,} rows")
        
        # Apply feature engineering
        df = apply_feature_engineering(df)
        
        # Clean data (remove NaN and sentinel values)
        required = ['aqi', 'aqi_level', 'windspeed', 'month', 'season']
        df = clean_data(df, required)
        
        # Encode categorical features
        df, encoders = encode_categorical_features(df)
        
        # Save to parquet (generates both full and model versions)
        save_cleaned_data(df, year=2023)
    
    if args.test or (not args.save and not args.test):
        # Test classification data preparation
        print("\n[Testing Classification Data Preparation]")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, encoders = prepare_classification_data()
            print(f"\nFeatures: {list(X_train.columns)}")
            print(f"Target classes: {encoders['aqi_level'].classes_}")
            print("\n✅ Classification data preparation successful!")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test regression data preparation
        print("\n[Testing Regression Data Preparation]")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test, encoders = prepare_regression_data()
            print(f"\nFeatures: {list(X_train.columns)}")
            print(f"Target range: {y_train.min():.1f} - {y_train.max():.1f}")
            print("\n✅ Regression data preparation successful!")
        except Exception as e:
            print(f"❌ Error: {e}")
