"""
Unit Tests for Data Preprocessing Module

Tests the feature engineering functions and data preparation utilities
for the air quality prediction models.

Run with: 
  From project root: python -m pytest src/main/python/models/test_data_preprocessing.py -v
  From models dir:   python -m pytest test_data_preprocessing.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the models directory to path for imports
MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(MODELS_DIR))

from data_preprocessing import (
    add_season,
    add_aqi_level,
    add_wind_level,
    split_data,
    encode_categorical_features,
    clean_data,
    create_lag_features,
    get_lag_feature_names,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'aqi': [30, 50, 51, 80, 100, 101, 120, 45, 55, 95, 105, 25],
        'windspeed': [0.5, 1.5, 1.6, 2.0, 3.4, 3.5, 5.0, 1.0, 2.5, 4.0, 0.8, 6.0],
        'county': ['新北市'] * 6 + ['彰化縣'] * 6
    })


@pytest.fixture
def df_with_nan():
    """Create DataFrame with missing values."""
    return pd.DataFrame({
        'month': [1, 2, 3, None, 5],
        'aqi': [30, None, 51, 80, 100],
        'windspeed': [0.5, 1.5, None, 2.0, 3.4],
    })


# ============================================================================
# Test: add_season()
# ============================================================================

class TestAddSeason:
    """Tests for season labeling function."""
    
    def test_winter_months(self, sample_df):
        """Test that months 12, 1, 2 are labeled as 冬季."""
        result = add_season(sample_df)
        
        # Month 1 (index 0) should be 冬季
        assert result.loc[0, 'season'] == '冬季'
        # Month 2 (index 1) should be 冬季
        assert result.loc[1, 'season'] == '冬季'
        # Month 12 (index 11) should be 冬季
        assert result.loc[11, 'season'] == '冬季'
    
    def test_spring_months(self, sample_df):
        """Test that months 3, 4, 5 are labeled as 春季."""
        result = add_season(sample_df)
        
        assert result.loc[2, 'season'] == '春季'  # Month 3
        assert result.loc[3, 'season'] == '春季'  # Month 4
        assert result.loc[4, 'season'] == '春季'  # Month 5
    
    def test_summer_months(self, sample_df):
        """Test that months 6, 7, 8 are labeled as 夏季."""
        result = add_season(sample_df)
        
        assert result.loc[5, 'season'] == '夏季'  # Month 6
        assert result.loc[6, 'season'] == '夏季'  # Month 7
        assert result.loc[7, 'season'] == '夏季'  # Month 8
    
    def test_autumn_months(self, sample_df):
        """Test that months 9, 10, 11 are labeled as 秋季."""
        result = add_season(sample_df)
        
        assert result.loc[8, 'season'] == '秋季'   # Month 9
        assert result.loc[9, 'season'] == '秋季'   # Month 10
        assert result.loc[10, 'season'] == '秋季'  # Month 11


# ============================================================================
# Test: add_aqi_level()
# ============================================================================

class TestAddAqiLevel:
    """Tests for AQI level classification function."""
    
    def test_good_level(self, sample_df):
        """Test AQI <= 50 is classified as 良好."""
        result = add_aqi_level(sample_df)
        
        # AQI 30 (index 0) should be 良好
        assert result.loc[0, 'aqi_level'] == '良好'
        # AQI 50 (index 1) should be 良好 (boundary)
        assert result.loc[1, 'aqi_level'] == '良好'
        # AQI 45 (index 7) should be 良好
        assert result.loc[7, 'aqi_level'] == '良好'
        # AQI 25 (index 11) should be 良好
        assert result.loc[11, 'aqi_level'] == '良好'
    
    def test_moderate_level(self, sample_df):
        """Test 50 < AQI <= 100 is classified as 普通."""
        result = add_aqi_level(sample_df)
        
        # AQI 51 (index 2) should be 普通
        assert result.loc[2, 'aqi_level'] == '普通'
        # AQI 80 (index 3) should be 普通
        assert result.loc[3, 'aqi_level'] == '普通'
        # AQI 100 (index 4) should be 普通 (boundary)
        assert result.loc[4, 'aqi_level'] == '普通'
        # AQI 95 (index 9) should be 普通
        assert result.loc[9, 'aqi_level'] == '普通'
    
    def test_unhealthy_level(self, sample_df):
        """Test AQI > 100 is classified as 對敏感族群不健康."""
        result = add_aqi_level(sample_df)
        
        # AQI 101 (index 5) should be 對敏感族群不健康 (boundary)
        assert result.loc[5, 'aqi_level'] == '對敏感族群不健康'
        # AQI 120 (index 6) should be 對敏感族群不健康
        assert result.loc[6, 'aqi_level'] == '對敏感族群不健康'
        # AQI 105 (index 10) should be 對敏感族群不健康
        assert result.loc[10, 'aqi_level'] == '對敏感族群不健康'
    
    def test_boundary_50(self):
        """Test exact boundary at AQI = 50."""
        df = pd.DataFrame({'aqi': [50]})
        result = add_aqi_level(df)
        assert result.loc[0, 'aqi_level'] == '良好'
    
    def test_boundary_100(self):
        """Test exact boundary at AQI = 100."""
        df = pd.DataFrame({'aqi': [100]})
        result = add_aqi_level(df)
        assert result.loc[0, 'aqi_level'] == '普通'
    
    def test_boundary_101(self):
        """Test just above boundary at AQI = 101."""
        df = pd.DataFrame({'aqi': [101]})
        result = add_aqi_level(df)
        assert result.loc[0, 'aqi_level'] == '對敏感族群不健康'


# ============================================================================
# Test: add_wind_level()
# ============================================================================

class TestAddWindLevel:
    """Tests for wind speed level classification function."""
    
    def test_no_wind(self, sample_df):
        """Test windspeed <= 1.5 is classified as 無風."""
        result = add_wind_level(sample_df)
        
        # windspeed 0.5 (index 0) should be 無風
        assert result.loc[0, 'wind_level'] == '無風'
        # windspeed 1.5 (index 1) should be 無風 (boundary)
        assert result.loc[1, 'wind_level'] == '無風'
        # windspeed 1.0 (index 7) should be 無風
        assert result.loc[7, 'wind_level'] == '無風'
        # windspeed 0.8 (index 10) should be 無風
        assert result.loc[10, 'wind_level'] == '無風'
    
    def test_light_wind(self, sample_df):
        """Test 1.5 < windspeed <= 3.4 is classified as 輕風."""
        result = add_wind_level(sample_df)
        
        # windspeed 1.6 (index 2) should be 輕風
        assert result.loc[2, 'wind_level'] == '輕風'
        # windspeed 2.0 (index 3) should be 輕風
        assert result.loc[3, 'wind_level'] == '輕風'
        # windspeed 3.4 (index 4) should be 輕風 (boundary)
        assert result.loc[4, 'wind_level'] == '輕風'
        # windspeed 2.5 (index 8) should be 輕風
        assert result.loc[8, 'wind_level'] == '輕風'
    
    def test_breeze(self, sample_df):
        """Test windspeed > 3.4 is classified as 微風以上."""
        result = add_wind_level(sample_df)
        
        # windspeed 3.5 (index 5) should be 微風以上
        assert result.loc[5, 'wind_level'] == '微風以上'
        # windspeed 5.0 (index 6) should be 微風以上
        assert result.loc[6, 'wind_level'] == '微風以上'
        # windspeed 4.0 (index 9) should be 微風以上
        assert result.loc[9, 'wind_level'] == '微風以上'
        # windspeed 6.0 (index 11) should be 微風以上
        assert result.loc[11, 'wind_level'] == '微風以上'


# ============================================================================
# Test: split_data()
# ============================================================================

class TestSplitData:
    """Tests for data splitting function."""
    
    def test_split_ratio(self):
        """Test that split ratios are approximately 70/15/15."""
        # Create larger dataset for meaningful ratio test
        n = 1000
        X = pd.DataFrame({'feature': np.random.randn(n)})
        y = pd.Series(np.random.randint(0, 3, n))
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        total = len(X_train) + len(X_val) + len(X_test)
        
        # Check train ratio (should be ~70% with 1% tolerance)
        train_ratio = len(X_train) / total
        assert 0.69 <= train_ratio <= 0.71, f"Train ratio {train_ratio:.2f} not in [0.69, 0.71]"
        
        # Check val ratio (should be ~15% with 1% tolerance)
        val_ratio = len(X_val) / total
        assert 0.14 <= val_ratio <= 0.16, f"Val ratio {val_ratio:.2f} not in [0.14, 0.16]"
        
        # Check test ratio (should be ~15% with 1% tolerance)
        test_ratio = len(X_test) / total
        assert 0.14 <= test_ratio <= 0.16, f"Test ratio {test_ratio:.2f} not in [0.14, 0.16]"
    
    def test_no_data_leakage(self):
        """Test that there is no overlap between train/val/test sets."""
        n = 100
        X = pd.DataFrame({'feature': range(n)})
        y = pd.Series(range(n))
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Check no overlap in indices
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        
        assert len(train_idx & val_idx) == 0, "Overlap between train and val"
        assert len(train_idx & test_idx) == 0, "Overlap between train and test"
        assert len(val_idx & test_idx) == 0, "Overlap between val and test"


# ============================================================================
# Test: clean_data() - No Missing Values
# ============================================================================

class TestCleanData:
    """Tests for data cleaning function."""
    
    def test_no_missing_values_after_clean(self, df_with_nan):
        """Test that no NaN values remain after cleaning."""
        required_cols = ['month', 'aqi', 'windspeed']
        result = clean_data(df_with_nan, required_cols)
        
        # Check no NaN in required columns
        for col in required_cols:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"NaN found in {col}"
    
    def test_rows_dropped(self, df_with_nan):
        """Test that rows with NaN are dropped."""
        required_cols = ['month', 'aqi', 'windspeed']
        result = clean_data(df_with_nan, required_cols)
        
        # Original has 5 rows, 3 have NaN values
        assert len(result) < len(df_with_nan)
    
    def test_sentinel_value_removed(self):
        """Test that AQI=-1 sentinel values are removed."""
        df = pd.DataFrame({
            'month': [1, 2, 3, 4, 5],
            'aqi': [-1, 50, -1, 100, 75],  # 2 sentinel values
            'windspeed': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        required_cols = ['month', 'aqi', 'windspeed']
        result = clean_data(df, required_cols)
        
        # Should have 3 rows left (removed 2 with AQI=-1)
        assert len(result) == 3
        # All remaining AQI should be >= 0
        assert (result['aqi'] >= 0).all()
    
    def test_no_negative_aqi_after_clean(self):
        """Test that no negative AQI values remain after cleaning."""
        df = pd.DataFrame({
            'month': [1, 2, 3, 4],
            'aqi': [-1, 0, 50, 100],  # -1 is sentinel, 0 is valid
            'windspeed': [1.0, 2.0, 3.0, 4.0]
        })
        required_cols = ['aqi']
        result = clean_data(df, required_cols)
        
        # AQI=0 should remain, AQI=-1 should be removed
        assert len(result) == 3
        assert -1 not in result['aqi'].values
        assert 0 in result['aqi'].values  # 0 is valid AQI
    
    def test_mixed_nan_and_sentinel(self):
        """Test cleaning with both NaN and sentinel values."""
        df = pd.DataFrame({
            'month': [1, 2, None, 4, 5],
            'aqi': [-1, 50, 75, None, 100],
            'windspeed': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        required_cols = ['month', 'aqi', 'windspeed']
        result = clean_data(df, required_cols)
        
        # Should remove: row 0 (AQI=-1), row 2 (month=NaN), row 3 (AQI=NaN)
        # Should keep: row 1 and row 4
        assert len(result) == 2
        assert (result['aqi'] >= 0).all()
        assert result['aqi'].isna().sum() == 0


# ============================================================================
# Test: encode_categorical_features()
# ============================================================================

class TestEncodeFeatures:
    """Tests for categorical feature encoding."""
    
    def test_encoded_columns_created(self, sample_df):
        """Test that encoded columns are created."""
        df = add_season(sample_df)
        result, encoders = encode_categorical_features(df, cat_cols=['season', 'county'])
        
        assert 'season_encoded' in result.columns
        assert 'county_encoded' in result.columns
    
    def test_encoded_values_are_integers(self, sample_df):
        """Test that encoded values are integers."""
        df = add_season(sample_df)
        result, encoders = encode_categorical_features(df, cat_cols=['season'])
        
        assert result['season_encoded'].dtype in [np.int32, np.int64, int]
    
    def test_encoders_returned(self, sample_df):
        """Test that encoders are returned for inverse transform."""
        df = add_season(sample_df)
        result, encoders = encode_categorical_features(df, cat_cols=['season'])
        
        assert 'season' in encoders
        assert hasattr(encoders['season'], 'inverse_transform')


# ============================================================================
# Test: create_lag_features()
# ============================================================================

class TestLagFeatures:
    """Tests for lag feature creation function."""
    
    @pytest.fixture
    def time_series_df(self):
        """Create a time-series DataFrame for testing lag features."""
        # 模擬兩個縣市各 5 小時的資料
        dates = pd.date_range('2023-01-01', periods=5, freq='h')
        return pd.DataFrame({
            'date': list(dates) * 2,
            'county': ['台北'] * 5 + ['高雄'] * 5,
            'pm2.5': [10, 20, 30, 40, 50] + [15, 25, 35, 45, 55],
            'pm10': [20, 30, 40, 50, 60] + [25, 35, 45, 55, 65],
            'o3': [5, 10, 15, 20, 25] + [8, 13, 18, 23, 28],
            'aqi': [30, 40, 50, 60, 70] + [35, 45, 55, 65, 75],
        })
    
    def test_lag_columns_created(self, time_series_df):
        """Test that lag columns are created correctly."""
        result = create_lag_features(time_series_df, lag_cols=['pm2.5', 'pm10'], lags=[1])
        
        assert 'pm2.5_lag1' in result.columns
        assert 'pm10_lag1' in result.columns
    
    def test_lag_shift_correct(self, time_series_df):
        """Test that lag1 equals the previous row's value within the same group."""
        result = create_lag_features(time_series_df, lag_cols=['pm2.5'], lags=[1], drop_na=False)
        
        # 檢查台北的 lag 值（排序後）
        taipei = result[result['county'] == '台北'].sort_values('date').reset_index(drop=True)
        
        # 第 0 筆應該是 NaN（沒有前一筆）
        assert pd.isna(taipei.loc[0, 'pm2.5_lag1'])
        
        # 第 1 筆的 lag1 應該等於第 0 筆的 pm2.5
        assert taipei.loc[1, 'pm2.5_lag1'] == 10  # 台北的第一個 pm2.5 值
        
        # 第 2 筆的 lag1 應該等於第 1 筆的 pm2.5
        assert taipei.loc[2, 'pm2.5_lag1'] == 20
    
    def test_group_by_county(self, time_series_df):
        """Test that shift is done within each county group (no cross-county leakage)."""
        result = create_lag_features(time_series_df, lag_cols=['pm2.5'], lags=[1], drop_na=False)
        
        # 高雄的第一筆也應該是 NaN（不是台北的最後一筆）
        kaohsiung_first = result[result['county'] == '高雄'].sort_values('date').iloc[0]
        assert pd.isna(kaohsiung_first['pm2.5_lag1'])
    
    def test_no_nan_after_dropna(self, time_series_df):
        """Test that no NaN remains after drop_na=True."""
        result = create_lag_features(time_series_df, lag_cols=['pm2.5'], lags=[1], drop_na=True)
        
        assert result['pm2.5_lag1'].isna().sum() == 0
    
    def test_rows_dropped_count(self, time_series_df):
        """Test that correct number of rows are dropped (1 per county for lag1)."""
        original_len = len(time_series_df)
        result = create_lag_features(time_series_df, lag_cols=['pm2.5'], lags=[1], drop_na=True)
        
        # 2 個縣市各 drop 1 筆 (因為 lag1 最前面 1 筆是 NaN)
        expected_dropped = 2  # 每個縣市 1 筆
        assert len(result) == original_len - expected_dropped
    
    def test_multiple_lags(self, time_series_df):
        """Test creating multiple lag steps."""
        result = create_lag_features(time_series_df, lag_cols=['pm2.5'], lags=[1, 2], drop_na=False)
        
        assert 'pm2.5_lag1' in result.columns
        assert 'pm2.5_lag2' in result.columns
    
    def test_get_lag_feature_names(self):
        """Test get_lag_feature_names helper function."""
        names = get_lag_feature_names(lag_cols=['pm2.5', 'pm10'], lags=[1])
        
        assert names == ['pm2.5_lag1', 'pm10_lag1']
    
    def test_get_lag_feature_names_multiple_lags(self):
        """Test get_lag_feature_names with multiple lags."""
        names = get_lag_feature_names(lag_cols=['pm2.5'], lags=[1, 24])
        
        assert names == ['pm2.5_lag1', 'pm2.5_lag24']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
