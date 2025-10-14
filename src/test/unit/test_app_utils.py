"""
Unit Tests for Application Utility Functions

Tests for app_utils.py module including:
- Data preparation and label generation
- Air quality structure calculations
- Filtering functions
- AQI color and recommendation functions

Author: Claude Code
Date: 2025-10-14
"""

import unittest
try:
    import pytest
    import pandas as pd
    import numpy as np
except Exception as e:
    raise unittest.SkipTest(f"Skipping test_app_utils due to missing dependencies: {e}")
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "main" / "python"
sys.path.insert(0, str(src_path))

from utils.app_utils import (
    prepare_data,
    空氣質量結構,
    get_aqi_color,
    get_aqi_recommendation,
    filter_data
)


@pytest.fixture
def sample_data():
    """
    Create sample air quality data for testing.

    Returns:
        DataFrame with sample air quality data
    """
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    n = len(dates)

    data = {
        'date': dates,
        'sitename': np.random.choice(['中山', '板橋', '桃園'], n),
        'county': np.random.choice(['台北市', '新北市', '桃園市'], n),
        'aqi': np.random.randint(20, 150, n),
        'pm2.5': np.random.uniform(5, 50, n),
        'pm10': np.random.uniform(10, 100, n),
        'o3': np.random.uniform(10, 80, n),
        'pollutant': np.random.choice(['PM2.5', 'PM10', 'O3'], n),
        'windspeed': np.random.uniform(0, 8, n),
        'latitude': np.random.uniform(24.5, 25.5, n),
        'longitude': np.random.uniform(120.5, 121.5, n)
    }

    return pd.DataFrame(data)


class TestPrepareData:
    """Test suite for prepare_data function"""

    def test_prepare_data_creates_time_labels(self, sample_data):
        """Test that time dimension labels are created correctly"""
        df = prepare_data(sample_data)

        # Check time columns exist
        assert 'year' in df.columns
        assert 'month' in df.columns
        assert 'day' in df.columns
        assert 'hour' in df.columns
        assert 'dayofweek' in df.columns
        assert 'quarter' in df.columns
        assert 'season' in df.columns
        assert 'yq' in df.columns
        assert 'ym' in df.columns

        # Check year value
        assert df['year'].iloc[0] == 2024

        # Check quarter labels
        assert df['quarter'].dtype == 'category'
        assert 'Q1' in df['quarter'].cat.categories

        # Check season labels (Chinese)
        assert df['season'].dtype == 'category'
        assert '春季' in df['season'].cat.categories

    def test_prepare_data_creates_space_labels(self, sample_data):
        """Test that space dimension labels are created correctly"""
        df = prepare_data(sample_data)

        # Check region column exists
        assert 'region' in df.columns

        # Check region mapping
        taipei_rows = df[df['county'] == '台北市']
        if len(taipei_rows) > 0:
            assert taipei_rows['region'].iloc[0] == '北部'

    def test_prepare_data_creates_pollutant_labels(self, sample_data):
        """Test that pollutant dimension labels are created correctly"""
        df = prepare_data(sample_data)

        # Check AQI level column
        assert 'aqi_level' in df.columns

        # Check AQI level classification
        good_aqi_rows = df[df['aqi'] <= 50]
        if len(good_aqi_rows) > 0:
            assert good_aqi_rows['aqi_level'].iloc[0] == '良好'

        # Check pollutant category
        assert 'pollutant_category' in df.columns

    def test_prepare_data_creates_condition_labels(self, sample_data):
        """Test that condition dimension labels are created correctly"""
        df = prepare_data(sample_data)

        # Check wind level column
        assert 'wind_level' in df.columns
        assert 'is_weekend' in df.columns
        assert 'time_period' in df.columns
        assert 'is_exceed' in df.columns

        # Check wind level categories
        assert df['wind_level'].dtype == 'category'

    def test_prepare_data_preserves_original_data(self, sample_data):
        """Test that original data is preserved after transformation"""
        original_len = len(sample_data)
        df = prepare_data(sample_data)

        assert len(df) == original_len
        assert 'aqi' in df.columns
        assert 'pm2.5' in df.columns


class TestAirQualityStructure:
    """Test suite for 空氣質量結構 function"""

    def test_structure_by_county(self, sample_data):
        """Test air quality structure calculation grouped by county"""
        df = prepare_data(sample_data)
        result = 空氣質量結構(df, 'county')

        # Check result columns
        assert 'county' in result.columns
        assert '監測站數' in result.columns
        assert '測量次數' in result.columns
        assert '平均AQI' in result.columns
        assert '達標率' in result.columns

        # Check result has correct number of groups
        assert len(result) == df['county'].nunique()

        # Check metrics are numeric
        assert pd.api.types.is_numeric_dtype(result['平均AQI'])
        assert pd.api.types.is_numeric_dtype(result['達標率'])

    def test_structure_calculates_correct_metrics(self, sample_data):
        """Test that metrics are calculated correctly"""
        df = prepare_data(sample_data)
        result = 空氣質量結構(df, 'county')

        # Check that station count makes sense
        assert (result['監測站數'] > 0).all()
        assert (result['測量次數'] > 0).all()

        # Check that compliance rate is percentage (0-100)
        assert (result['達標率'] >= 0).all()
        assert (result['達標率'] <= 100).all()

    def test_structure_with_different_grouping(self, sample_data):
        """Test structure calculation with different grouping columns"""
        df = prepare_data(sample_data)

        # Test with region grouping
        result = 空氣質量結構(df, 'region')
        assert 'region' in result.columns
        assert len(result) == df['region'].nunique()


class TestGetAQIColor:
    """Test suite for get_aqi_color function"""

    def test_aqi_color_good(self):
        """Test color for good AQI (0-50)"""
        color = get_aqi_color(30)
        assert color == "#00E400"

    def test_aqi_color_moderate(self):
        """Test color for moderate AQI (51-100)"""
        color = get_aqi_color(75)
        assert color == "#FFFF00"

    def test_aqi_color_unhealthy_sensitive(self):
        """Test color for unhealthy for sensitive groups (101-150)"""
        color = get_aqi_color(120)
        assert color == "#FF7E00"

    def test_aqi_color_unhealthy(self):
        """Test color for unhealthy (151-200)"""
        color = get_aqi_color(175)
        assert color == "#FF0000"

    def test_aqi_color_very_unhealthy(self):
        """Test color for very unhealthy (201-300)"""
        color = get_aqi_color(250)
        assert color == "#8F3F97"

    def test_aqi_color_hazardous(self):
        """Test color for hazardous (301+)"""
        color = get_aqi_color(350)
        assert color == "#7E0023"

    def test_aqi_color_nan(self):
        """Test color for NaN value"""
        color = get_aqi_color(np.nan)
        assert color == "#CCCCCC"


class TestGetAQIRecommendation:
    """Test suite for get_aqi_recommendation function"""

    def test_recommendation_general_good(self):
        """Test recommendation for general public with good AQI"""
        advice = get_aqi_recommendation(40, "一般民眾")
        assert "良好" in advice or "適合" in advice

    def test_recommendation_general_unhealthy(self):
        """Test recommendation for general public with unhealthy AQI"""
        advice = get_aqi_recommendation(160, "一般民眾")
        assert "減少" in advice or "口罩" in advice

    def test_recommendation_sensitive_moderate(self):
        """Test recommendation for sensitive groups with moderate AQI"""
        advice = get_aqi_recommendation(75, "敏感族群")
        assert len(advice) > 0

    def test_recommendation_outdoor_worker(self):
        """Test recommendation for outdoor workers"""
        advice = get_aqi_recommendation(120, "戶外工作者")
        assert len(advice) > 0

    def test_recommendation_athlete(self):
        """Test recommendation for athletes"""
        advice = get_aqi_recommendation(90, "運動愛好者")
        assert len(advice) > 0

    def test_recommendation_nan(self):
        """Test recommendation with NaN AQI"""
        advice = get_aqi_recommendation(np.nan, "一般民眾")
        assert "異常" in advice


class TestFilterData:
    """Test suite for filter_data function"""

    def test_filter_by_date_range(self, sample_data):
        """Test filtering by date range"""
        filtered = filter_data(
            sample_data,
            start_date='2024-01-15',
            end_date='2024-01-20'
        )

        assert len(filtered) < len(sample_data)
        assert filtered['date'].min() >= pd.to_datetime('2024-01-15')
        assert filtered['date'].max() <= pd.to_datetime('2024-01-20')

    def test_filter_by_counties(self, sample_data):
        """Test filtering by counties"""
        filtered = filter_data(
            sample_data,
            counties=['台北市']
        )

        assert len(filtered) <= len(sample_data)
        assert (filtered['county'] == '台北市').all()

    def test_filter_by_stations(self, sample_data):
        """Test filtering by stations"""
        filtered = filter_data(
            sample_data,
            stations=['中山', '板橋']
        )

        assert len(filtered) <= len(sample_data)
        assert filtered['sitename'].isin(['中山', '板橋']).all()

    def test_filter_combined(self, sample_data):
        """Test filtering with multiple criteria"""
        filtered = filter_data(
            sample_data,
            start_date='2024-01-15',
            end_date='2024-01-20',
            counties=['台北市', '新北市']
        )

        assert len(filtered) <= len(sample_data)
        assert filtered['date'].min() >= pd.to_datetime('2024-01-15')
        assert filtered['county'].isin(['台北市', '新北市']).all()

    def test_filter_no_criteria(self, sample_data):
        """Test that no filtering returns full dataset"""
        filtered = filter_data(sample_data)
        assert len(filtered) == len(sample_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
