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
from typing import Tuple, Optional, List, Dict, Any
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
# Lag Features (避免 Data Leakage)
# ============================================================================

# 預設使用的污染物欄位及滯後步數
DEFAULT_LAG_COLS = ['pm2.5', 'pm10', 'o3']
DEFAULT_LAGS = [1]  # 只用 lag1 (前一小時)


def create_lag_features(
    df: pd.DataFrame,
    lag_cols: List[str] = None,
    lags: List[int] = None,
    group_col: str = 'county',
    date_col: str = 'date',
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create lag features for time-series prediction.
    
    為污染物欄位產生滯後特徵，避免 Data Leakage。
    例如：用 t-1 時刻的 PM2.5 預測 t 時刻的 AQI。
    
    ⚠️ 重要：
    - 資料必須先按 (group_col, date_col) 排序
    - shift() 按 group 分組，避免不同縣市的資料混淆
    - 產生的 lag 欄位前幾筆會是 NaN（因為沒有前一小時資料）
    
    Args:
        df: DataFrame with pollutant columns
        lag_cols: Columns to create lag features for (default: pm2.5, pm10, o3)
        lags: List of lag steps (default: [1] = 前一小時)
        group_col: Column to group by (prevent cross-county leakage)
        date_col: Column for sorting by time
        drop_na: Whether to drop rows with NaN lag values
        
    Returns:
        DataFrame with lag features added (e.g., pm2.5_lag1, pm10_lag1)
        
    Example:
        >>> df = create_lag_features(df, lag_cols=['pm2.5', 'pm10'], lags=[1, 24])
        # Creates: pm2.5_lag1, pm2.5_lag24, pm10_lag1, pm10_lag24
    """
    if lag_cols is None:
        lag_cols = DEFAULT_LAG_COLS
    if lags is None:
        lags = DEFAULT_LAGS
    
    df = df.copy()
    
    # 確保資料按時間排序
    if date_col in df.columns:
        df = df.sort_values([group_col, date_col]).reset_index(drop=True)
    
    # 產生 lag 特徵
    lag_feature_names = []
    for col in lag_cols:
        if col not in df.columns:
            print(f"  Warning: Column '{col}' not found, skipping lag features for it.")
            continue
            
        for lag in lags:
            lag_col_name = f'{col}_lag{lag}'
            # groupby 確保不同縣市不會互相 shift
            df[lag_col_name] = df.groupby(group_col)[col].shift(lag)
            lag_feature_names.append(lag_col_name)
    
    created_count = len(lag_feature_names)
    print(f"Created {created_count} lag features: {lag_feature_names}")
    
    # 移除因 shift 產生的 NaN
    if drop_na and lag_feature_names:
        before_len = len(df)
        df = df.dropna(subset=lag_feature_names)
        dropped = before_len - len(df)
        if dropped > 0:
            print(f"Dropped {dropped:,} rows with NaN lag values ({dropped/before_len*100:.2f}%)")
    
    return df


def get_lag_feature_names(lag_cols: List[str] = None, lags: List[int] = None) -> List[str]:
    """
    Get the list of lag feature column names that would be created.
    
    Args:
        lag_cols: Columns to create lag features for
        lags: List of lag steps
        
    Returns:
        List of lag feature column names
    """
    if lag_cols is None:
        lag_cols = DEFAULT_LAG_COLS
    if lags is None:
        lags = DEFAULT_LAGS
    
    return [f'{col}_lag{lag}' for col in lag_cols for lag in lags]


# ============================================================================
# Class Distribution Check (FR-001-D: Phase 1 觀察用)
# ============================================================================

def check_class_distribution(
    df: pd.DataFrame,
    target_col: str = 'aqi_level',
    verbose: bool = True,
    imbalance_threshold: float = 0.10
) -> Dict[str, Any]:
    """
    檢查並報告類別分佈情況（Phase 1: 僅觀察，不做處理）。
    
    此函式用於診斷類別不平衡問題，實際處理在 Phase 3/4 模型訓練時進行
    （使用 class_weight='balanced' 或 SMOTE）。
    
    Args:
        df: DataFrame with target column
        target_col: Name of target column (default: 'aqi_level')
        verbose: Whether to print detailed report (default: True)
        imbalance_threshold: Threshold below which a class is considered minority (default: 0.10 = 10%)
        
    Returns:
        Dict with:
        - 'counts': 各類別數量 (pd.Series)
        - 'percentages': 各類別百分比 (pd.Series)
        - 'is_imbalanced': 是否有嚴重不平衡（任一類別 < threshold）
        - 'minority_classes': 少數類別列表（百分比 < threshold）
        - 'majority_class': 多數類別名稱
        - 'imbalance_ratio': 多數類別 / 少數類別的比例
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame")
    
    # Calculate distribution
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True)
    
    # Detect imbalance
    minority_classes = percentages[percentages < imbalance_threshold].index.tolist()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    imbalance_ratio = counts[majority_class] / counts[minority_class] if counts[minority_class] > 0 else float('inf')
    
    is_imbalanced = len(minority_classes) > 0
    
    result = {
        'counts': counts,
        'percentages': percentages,
        'is_imbalanced': is_imbalanced,
        'minority_classes': minority_classes,
        'majority_class': majority_class,
        'imbalance_ratio': imbalance_ratio
    }
    
    # Print report
    if verbose:
        print("\n" + "=" * 50)
        print(f"📊 Class Distribution Report: '{target_col}'")
        print("=" * 50)
        print(f"Total samples: {len(df):,}")
        print()
        print("Class Distribution:")
        for cls in counts.index:
            pct = percentages[cls] * 100
            bar = "█" * int(pct / 2)  # Visual bar (50 chars max)
            marker = " ⚠️ MINORITY" if cls in minority_classes else ""
            print(f"  {cls:20s}: {counts[cls]:>8,} ({pct:5.1f}%) {bar}{marker}")
        print()
        
        if is_imbalanced:
            print(f"⚠️  WARNING: Class imbalance detected!")
            print(f"   Minority classes (< {imbalance_threshold*100:.0f}%): {minority_classes}")
            print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1 ({majority_class} vs {minority_class})")
            print(f"   💡 Recommendation: Use class_weight='balanced' in Phase 3/4")
        else:
            print("✅ No severe class imbalance detected.")
        
        print("=" * 50 + "\n")
    
    return result


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
    'season',         # 原始季節 (供參考)
    # One-Hot Encoded Seasons (取代 season_encoded)
    'season_spring', 'season_summer', 'season_autumn', 'season_winter',
    'county_encoded',                 # 空間
    'wind_level', 'wind_level_encoded',  # 風速等級
    # Lag Features (避免 Data Leakage)
    'pm2.5', 'pm10', 'o3',           # 原始污染物（供產生 lag 用）
    'pm2.5_lag1', 'pm10_lag1', 'o3_lag1',  # 滯後特徵
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


def save_multiyear_splits(
    train_years: List[int] = None,
    test_year: int = 2023,
    counties: Optional[List[str]] = None
) -> dict:
    """
    Save pre-processed train/val/test splits for multi-year training.
    
    產出三個檔案：
    - train_{start}_{end}.parquet: 訓練集（多年資料）
    - val_{test_year}_h1.parquet: 驗證集（測試年上半年）
    - test_{test_year}_h2.parquet: 測試集（測試年下半年）
    
    Args:
        train_years: Years for training (default: 2017-2022)
        test_year: Year for validation/testing (default: 2023)
        counties: Counties to include
        
    Returns:
        Dict with paths to all saved files
    """
    if train_years is None:
        train_years = list(range(2017, 2023))
    if counties is None:
        counties = ['New Taipei City', 'Changhua County', 'Kaohsiung City']
    
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Multi-Year Training Splits")
    print("=" * 60)
    
    # Load and process training data
    print(f"\n[1/4] Loading training data ({train_years[0]}-{train_years[-1]})...")
    df_train = load_multi_year_data(years=train_years, counties=counties)
    df_train = apply_feature_engineering(df_train)
    
    # Create lag features (避免 Data Leakage)
    print(f"\n[2/4] Creating lag features...")
    df_train = create_lag_features(df_train)
    
    required = ['aqi', 'aqi_level', 'windspeed', 'month', 'date', 'season']
    df_train = clean_data(df_train, required)
    df_train, encoders = encode_categorical_features(df_train)
    # One-Hot Encoding for season
    df_train = encode_season_onehot(df_train)
    
    # Load and process test year data
    print(f"\n[3/4] Loading test year data ({test_year})...")
    df_test_year = load_training_data(year=test_year, counties=counties)
    df_test_year = apply_feature_engineering(df_test_year)
    df_test_year = create_lag_features(df_test_year)
    df_test_year = clean_data(df_test_year, required)
    
    # Apply same encoding (fit on train)
    for col, le in encoders.items():
        if col in df_test_year.columns:
            df_test_year[f'{col}_encoded'] = df_test_year[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
            
    # One-Hot Encoding for season
    df_test_year = encode_season_onehot(df_test_year)
    
    # Sort and split test year
    df_test_year = df_test_year.sort_values('date').reset_index(drop=True)
    mid_point = len(df_test_year) // 2
    df_val = df_test_year.iloc[:mid_point]
    df_test = df_test_year.iloc[mid_point:]
    
    # Select only model columns
    available_cols = [col for col in MODEL_COLS if col in df_train.columns]
    df_train = df_train[available_cols]
    df_val = df_val[available_cols]
    df_test = df_test[available_cols]
    
    # Save files
    print(f"\n[4/4] Saving splits...")
    results = {}
    
    # Train
    train_path = CLEANED_DATA_DIR / f'train_{train_years[0]}_{train_years[-1]}.parquet'
    df_train.to_parquet(train_path, index=False, compression='snappy')
    results['train'] = train_path
    print(f"  ✅ Train: {train_path.name} ({len(df_train):,} rows, {train_path.stat().st_size/1024/1024:.1f} MB)")
    
    # Validation
    val_path = CLEANED_DATA_DIR / f'val_{test_year}_h1.parquet'
    df_val.to_parquet(val_path, index=False, compression='snappy')
    results['val'] = val_path
    print(f"  ✅ Val:   {val_path.name} ({len(df_val):,} rows, {val_path.stat().st_size/1024/1024:.1f} MB)")
    
    # Test
    test_path = CLEANED_DATA_DIR / f'test_{test_year}_h2.parquet'
    df_test.to_parquet(test_path, index=False, compression='snappy')
    results['test'] = test_path
    print(f"  ✅ Test:  {test_path.name} ({len(df_test):,} rows, {test_path.stat().st_size/1024/1024:.1f} MB)")
    
    print(f"\n✅ All splits saved to: {CLEANED_DATA_DIR}")
    
    return results


def load_training_splits(
    train_years: List[int] = None,
    test_year: int = 2023
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-saved train/val/test splits.
    
    Args:
        train_years: Training year range (for filename)
        test_year: Test year (for filename)
        
    Returns:
        Tuple of (df_train, df_val, df_test)
    """
    if train_years is None:
        train_years = list(range(2017, 2023))
    
    train_path = CLEANED_DATA_DIR / f'train_{train_years[0]}_{train_years[-1]}.parquet'
    val_path = CLEANED_DATA_DIR / f'val_{test_year}_h1.parquet'
    test_path = CLEANED_DATA_DIR / f'test_{test_year}_h2.parquet'
    
    # Check all files exist
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}\n"
                f"Run save_multiyear_splits() first to generate splits."
            )
    
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    print(f"✅ Loaded training splits:")
    print(f"   Train: {len(df_train):,} rows")
    print(f"   Val:   {len(df_val):,} rows")
    print(f"   Test:  {len(df_test):,} rows")
    
    return df_train, df_val, df_test


def splits_exist(train_years: List[int] = None, test_year: int = 2023) -> bool:
    """Check if all split files exist."""
    if train_years is None:
        train_years = list(range(2017, 2023))
    
    train_path = CLEANED_DATA_DIR / f'train_{train_years[0]}_{train_years[-1]}.parquet'
    val_path = CLEANED_DATA_DIR / f'val_{test_year}_h1.parquet'
    test_path = CLEANED_DATA_DIR / f'test_{test_year}_h2.parquet'
    
    return all(p.exists() for p in [train_path, val_path, test_path])


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def load_training_data(
    year: int = 2023,
    counties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load training data from parquet files for a single year.
    
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


def load_multi_year_data(
    years: List[int],
    counties: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load data from multiple years for comprehensive training.
    
    多年訓練方案：讓模型學習所有季節的完整模式。
    例如：用 2016-2022 年資料訓練，模型可學到 7 年份 × 12 月份的季節規律。
    
    Args:
        years: List of years to load (e.g., [2016, 2017, ..., 2022])
        counties: List of counties to filter
        
    Returns:
        Combined DataFrame from all specified years
    """
    if counties is None:
        counties = ['New Taipei City', 'Changhua County', 'Kaohsiung City']
    
    loader = AirQualityDataLoader(data_dir=str(PROJECT_ROOT))
    
    all_dfs = []
    for year in years:
        try:
            df = loader.load_by_year(year)
            if counties:
                df = df[df['county'].isin(counties)]
            all_dfs.append(df)
            print(f"  Loaded {year}: {len(df):,} rows")
        except Exception as e:
            print(f"  Warning: Could not load {year}: {e}")
    
    loader.close()
    
    if not all_dfs:
        raise ValueError("No data loaded from any year")
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined_df):,} rows from {len(all_dfs)} years")
    
    return combined_df


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



def encode_season_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot encode season column manually.
    
    Creates 4 binary columns:
    - season_spring
    - season_summer
    - season_autumn
    - season_winter
    
    Args:
        df: DataFrame with 'season' column
        
    Returns:
        DataFrame with one-hot encoded columns added
    """
    df = df.copy()
    # Map Chinese season names to English suffixes
    season_map = {
        '春季': 'spring',
        '夏季': 'summer',
        '秋季': 'autumn',
        '冬季': 'winter'
    }
    
    # Ensure season column exists
    if 'season' not in df.columns:
        print("Warning: 'season' column not found, skipping one-hot encoding.")
        return df
        
    for ch_name, en_suffix in season_map.items():
        # Create binary column (1 if match, 0 if not)
        df[f'season_{en_suffix}'] = (df['season'] == ch_name).astype(int)
        
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    cat_cols: List[str] = ['county', 'wind_level']
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


def split_data_temporal(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    date_col: str = 'date',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data chronologically for time-series to prevent data leakage.
    
    ⚠️ 重要：時間序列資料必須按時間順序分割！
    隨機分割會導致「用未來預測過去」的資料洩漏問題。
    
    分割策略（以 2023 年為例）：
    - 訓練集 (70%): 1月 - 9月初
    - 驗證集 (15%): 9月初 - 11月初  
    - 測試集 (15%): 11月初 - 12月底
    
    Args:
        df: DataFrame with date column (must be sorted or will be sorted)
        target_col: Name of target column
        feature_cols: List of feature column names
        date_col: Name of date column for sorting
        train_ratio: Proportion for training set (default: 0.70)
        val_ratio: Proportion for validation set (default: 0.15)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Ensure data is sorted by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Split by position (chronological order)
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Extract features and target
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    
    # Report date ranges
    if date_col in df.columns:
        train_dates = f"{train_df[date_col].min()} ~ {train_df[date_col].max()}"
        val_dates = f"{val_df[date_col].min()} ~ {val_df[date_col].max()}"
        test_dates = f"{test_df[date_col].min()} ~ {test_df[date_col].max()}"
        
        print(f"Temporal data split complete (chronological order):")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%) | {train_dates}")
        print(f"  Val:   {len(X_val):,} samples ({len(X_val)/n*100:.1f}%) | {val_dates}")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/n*100:.1f}%) | {test_dates}")
    else:
        print(f"Temporal data split complete:")
        print(f"  Train: {len(X_train):,} samples ({len(X_train)/n*100:.1f}%)")
        print(f"  Val:   {len(X_val):,} samples ({len(X_val)/n*100:.1f}%)")
        print(f"  Test:  {len(X_test):,} samples ({len(X_test)/n*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets (70/15/15) using RANDOM splitting.
    
    ⚠️ 注意：此函式使用隨機分割，僅適用於非時間序列資料！
    對於時間序列資料（如空氣品質預測），請使用 split_data_temporal()。
    
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
    
    print(f"Random data split complete (WARNING: not suitable for time-series!):")
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
    
    使用時間序列分割（按照日期順序），避免資料洩漏。
    
    Args:
        year: Year of data to use
        counties: Counties to include
        feature_cols: Feature columns (default: windspeed, month, hour, season_encoded, county_encoded)
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, encoders)
    """
    if feature_cols is None:
        feature_cols = ['windspeed', 'month', 'hour', 'season_encoded', 'county_encoded']
    
    # Load and prepare data
    df = load_training_data(year=year, counties=counties)
    df = apply_feature_engineering(df)
    
    # Clean data
    required = ['aqi', 'windspeed', 'month', 'date']
    df = clean_data(df, required)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Get available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Split data using TEMPORAL splitting (chronological order)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_temporal(
        df=df,
        target_col='aqi',
        feature_cols=available_features,
        date_col='date'
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


def prepare_classification_data(
    year: int = 2023,
    counties: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for classification task (predicting AQI level).
    
    使用時間序列分割（按照日期順序），避免資料洩漏。
    
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
    required = ['aqi', 'aqi_level', 'windspeed', 'month', 'date']
    df = clean_data(df, required)
    
    # Encode categorical features
    df, encoders = encode_categorical_features(df)
    
    # Encode target variable (add encoded column to df for temporal split)
    le_target = LabelEncoder()
    df['aqi_level_encoded'] = le_target.fit_transform(df['aqi_level'])
    encoders['aqi_level'] = le_target
    
    # Get available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Split data using TEMPORAL splitting (chronological order)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_temporal(
        df=df,
        target_col='aqi_level_encoded',
        feature_cols=available_features,
        date_col='date'
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


def prepare_regression_data_multiyear(
    train_years: List[int] = None,
    test_year: int = 2023,
    counties: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for regression using MULTI-YEAR training approach.
    
    多年訓練方案（最佳實踐）：
    - 訓練集：2017-2022 年（6年 × 12月 = 完整季節學習）
    - 驗證集：2023 年 1-6 月
    - 測試集：2023 年 7-12 月
    
    優點：
    1. 模型學習所有月份的季節模式
    2. 沒有時間洩漏（永遠用過去預測未來）
    3. 測試評估結果真實可信
    
    Args:
        train_years: Years for training (default: 2017-2022)
        test_year: Year for validation and testing (default: 2023)
        counties: Counties to include
        feature_cols: Feature columns
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, encoders)
    """
    if train_years is None:
        train_years = list(range(2017, 2023))  # 2017-2022
    if feature_cols is None:
        # 預設特徵包含 lag 特徵（避免 Data Leakage）
        feature_cols = [
            'pm2.5_lag1', 'pm10_lag1', 'o3_lag1',  # Lag 特徵
            'windspeed', 'month', 'hour',          # 氣象 & 時間
            'season_encoded', 'county_encoded'     # 編碼
        ]
    
    print(f"[Multi-Year Training] Train: {train_years}, Val/Test: {test_year}")
    
    # Load training data (multiple years)
    print("Loading training data...")
    df_train = load_multi_year_data(years=train_years, counties=counties)
    
    # Load test year data
    print(f"Loading test year data ({test_year})...")
    df_test_year = load_training_data(year=test_year, counties=counties)
    
    # Apply feature engineering to both
    df_train = apply_feature_engineering(df_train)
    df_test_year = apply_feature_engineering(df_test_year)
    
    # Create lag features (避免 Data Leakage)
    print("Creating lag features...")
    df_train = create_lag_features(df_train)
    df_test_year = create_lag_features(df_test_year)
    
    # Clean data (lag features 已經在 create_lag_features 處理 NaN)
    required = ['aqi', 'windspeed', 'month', 'date']
    df_train = clean_data(df_train, required)
    df_test_year = clean_data(df_test_year, required)
    
    # Encode categorical features (fit on train, transform both)
    df_train, encoders = encode_categorical_features(df_train)
    
    # Apply same encoding to test year data
    for col, le in encoders.items():
        if col in df_test_year.columns:
            # Handle unseen categories
            df_test_year[f'{col}_encoded'] = df_test_year[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Sort test year by date
    df_test_year = df_test_year.sort_values('date').reset_index(drop=True)
    
    # Split test year into validation (Jan-Jun) and test (Jul-Dec)
    mid_point = len(df_test_year) // 2
    df_val = df_test_year.iloc[:mid_point]
    df_test = df_test_year.iloc[mid_point:]
    
    # Get available features
    available_features = [col for col in feature_cols if col in df_train.columns]
    
    # Extract X and y
    X_train = df_train[available_features]
    y_train = df_train['aqi']
    
    X_val = df_val[available_features]
    y_val = df_val['aqi']
    
    X_test = df_test[available_features]
    y_test = df_test['aqi']
    
    # Report splits
    print(f"\nMulti-Year Training Split Complete:")
    print(f"  Train: {len(X_train):,} samples | {train_years[0]}-{train_years[-1]} (all months)")
    print(f"  Val:   {len(X_val):,} samples | {test_year} (first half)")
    print(f"  Test:  {len(X_test):,} samples | {test_year} (second half)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, encoders


def prepare_classification_data_multiyear(
    train_years: List[int] = None,
    test_year: int = 2023,
    counties: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """
    Prepare data for classification using MULTI-YEAR training approach.
    
    多年訓練方案（最佳實踐）：
    - 訓練集：2017-2022 年（6年 × 12月 = 完整季節學習）
    - 驗證集：2023 年 1-6 月
    - 測試集：2023 年 7-12 月
    
    Args:
        train_years: Years for training (default: 2017-2022)
        test_year: Year for validation and testing (default: 2023)
        counties: Counties to include
        feature_cols: Feature columns
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, encoders)
    """
    if train_years is None:
        train_years = list(range(2017, 2023))  # 2017-2022
    if feature_cols is None:
        # 預設特徵包含 lag 特徵（避免 Data Leakage）
        feature_cols = [
            'pm2.5_lag1', 'pm10_lag1', 'o3_lag1',  # Lag 特徵
            'windspeed', 'month', 'hour',          # 氣象 & 時間
            'season_encoded', 'county_encoded'     # 編碼
        ]
    
    print(f"[Multi-Year Training] Train: {train_years}, Val/Test: {test_year}")
    
    # Load training data (multiple years)
    print("Loading training data...")
    df_train = load_multi_year_data(years=train_years, counties=counties)
    
    # Load test year data
    print(f"Loading test year data ({test_year})...")
    df_test_year = load_training_data(year=test_year, counties=counties)
    
    # Apply feature engineering to both
    df_train = apply_feature_engineering(df_train)
    df_test_year = apply_feature_engineering(df_test_year)
    
    # Create lag features (避免 Data Leakage)
    print("Creating lag features...")
    df_train = create_lag_features(df_train)
    df_test_year = create_lag_features(df_test_year)
    
    # Clean data (lag features 已經在 create_lag_features 處理 NaN)
    required = ['aqi', 'aqi_level', 'windspeed', 'month', 'date']
    df_train = clean_data(df_train, required)
    df_test_year = clean_data(df_test_year, required)
    
    # Encode categorical features (fit on train)
    df_train, encoders = encode_categorical_features(df_train)
    
    # Encode target variable
    le_target = LabelEncoder()
    df_train['aqi_level_encoded'] = le_target.fit_transform(df_train['aqi_level'])
    encoders['aqi_level'] = le_target
    
    # Apply same encoding to test year data
    for col, le in encoders.items():
        if col == 'aqi_level':
            df_test_year['aqi_level_encoded'] = df_test_year['aqi_level'].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        elif col in df_test_year.columns:
            df_test_year[f'{col}_encoded'] = df_test_year[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Sort test year by date
    df_test_year = df_test_year.sort_values('date').reset_index(drop=True)
    
    # Split test year into validation (Jan-Jun) and test (Jul-Dec)
    mid_point = len(df_test_year) // 2
    df_val = df_test_year.iloc[:mid_point]
    df_test = df_test_year.iloc[mid_point:]
    
    # Get available features
    available_features = [col for col in feature_cols if col in df_train.columns]
    
    # Extract X and y
    X_train = df_train[available_features]
    y_train = df_train['aqi_level_encoded']
    
    X_val = df_val[available_features]
    y_val = df_val['aqi_level_encoded']
    
    X_test = df_test[available_features]
    y_test = df_test['aqi_level_encoded']
    
    # Report splits
    print(f"\nMulti-Year Training Split Complete:")
    print(f"  Train: {len(X_train):,} samples | {train_years[0]}-{train_years[-1]} (all months)")
    print(f"  Val:   {len(X_val):,} samples | {test_year} (first half)")
    print(f"  Test:  {len(X_test):,} samples | {test_year} (second half)")
    print(f"  Classes: {le_target.classes_}")
    
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
