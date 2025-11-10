"""
Application Utility Functions for Air Quality Streamlit App

This module provides core utility functions for the Streamlit application including:
- Session state initialization and management
- Data preparation and SPCT dimension label generation
- Air quality structure calculations (similar to sales structure)
- Common aggregation and filtering functions
- Operation logging

Author: Claude Code
Date: 2025-10-14
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state(project_name: str = "空氣質量分析系統") -> Any:
    """
    Initialize all session state variables for the Streamlit application.

    This function sets up persistent variables that maintain state across
    page navigations and user interactions.

    Args:
        project_name: Name of the project for logging purposes

    Returns:
        Streamlit session_state object with initialized variables

    Example:
        >>> sss = init_session_state("Air Quality Analysis")
        >>> sss.df is None
        True
    """
    sss = st.session_state

    # Data storage
    if 'df' not in sss:
        sss.df = None
    if 'df_filtered' not in sss:
        sss.df_filtered = None

    # User selections - data filters
    if 'selected_counties' not in sss:
        sss.selected_counties = []
    if 'selected_stations' not in sss:
        sss.selected_stations = []
    if 'selected_pollutants' not in sss:
        sss.selected_pollutants = ['PM2.5', 'PM10']
    if 'date_range' not in sss:
        sss.date_range = None

    # Analysis parameters
    if 'stat_method' not in sss:
        sss.stat_method = "平均值"
    if 'time_agg' not in sss:
        sss.time_agg = "日"
    if 'aqi_threshold' not in sss:
        sss.aqi_threshold = 100

    # Analysis results cache
    if 'analysis_result' not in sss:
        sss.analysis_result = None
    if 'crosstab_result' not in sss:
        sss.crosstab_result = None

    # Operation log
    if 'log' not in sss:
        sss.log = [f"[{datetime.now().strftime('%H:%M:%S')}] {project_name} 初始化"]

    # Note: user input fields removed per requirements (no user feature)

    return sss


def add_log(message: str, log_list: Optional[List[str]] = None) -> None:
    """
    Add timestamped entry to operation log.

    Args:
        message: Log message to add
        log_list: Optional log list (defaults to session_state.log)

    Example:
        >>> add_log("載入數據完成")
        >>> st.session_state.log[-1]
        '[14:30:45] 載入數據完成'
    """
    if log_list is None:
        log_list = st.session_state.log

    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_list.append(log_entry)
    logger.info(message)


def categorize_windspeed(windspeed: float) -> str:
    """
    Categorize wind speed into Beaufort scale levels (meters per second).

    This unified function ensures consistent wind level classification across
    all pages and visualizations in the application.

    Wind Level Standards (Beaufort Scale adapted for m/s):
        - 無風 (Calm): 0 - 1.5 m/s
        - 輕風 (Light air): 1.6 - 3.3 m/s
        - 微風 (Light breeze): 3.4 - 5.4 m/s
        - 和風 (Gentle breeze): 5.5 - 7.9 m/s
        - 強風 (Moderate breeze): ≥ 8.0 m/s

    Args:
        windspeed: Wind speed value in meters per second

    Returns:
        String label representing the wind level category

    Example:
        >>> categorize_windspeed(1.2)
        '無風(0-1.5)'
        >>> categorize_windspeed(2.5)
        '輕風(1.6-3.3)'
        >>> categorize_windspeed(10.0)
        '強風(≥8.0)'

    Note:
        This function is the single source of truth for wind level classification.
        All pages (page2, page3, etc.) should use this function to ensure consistency.
    """
    if pd.isna(windspeed):
        return None
    elif windspeed <= 1.5:
        return '無風(0-1.5)'
    elif windspeed <= 3.3:
        return '輕風(1.6-3.3)'
    elif windspeed <= 5.4:
        return '微風(3.4-5.4)'
    elif windspeed <= 7.9:
        return '和風(5.5-7.9)'
    else:
        return '強風(≥8.0)'


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data by generating SPCT dimension labels and derived attributes.

    This function implements the data transformation described in PLANNING.md
    for the SPCT model (Space, Pollutant, Condition, Time dimensions).

    Args:
        df: Raw air quality DataFrame

    Returns:
        DataFrame with additional dimension labels

    Dimension Labels Generated:
        - Time (T): year, month, day, hour, dayofweek, quarter, season, yq, ym
        - Space (S): region (derived from county)
        - Pollutant (P): pollutant_category, aqi_level
        - Condition (C): wind_level, time_period, is_weekend
    """
    df = df.copy()

    # ===== Time Dimension (T) =====
    # Convert date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Extract time components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek

    # Quarter labels
    df['quarter'] = pd.cut(df['month'],
                           bins=[0, 3, 6, 9, 12],
                           labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Season labels (Chinese)
    df['season'] = pd.cut(df['month'],
                          bins=[0, 3, 6, 9, 12],
                          labels=['冬季', '春季', '夏季', '秋季'])

    # Year-quarter combination (e.g., "24Q3")
    df['yq'] = df['year'].astype(str).str[-2:] + df['quarter'].astype(str)

    # Year-month combination
    df['ym'] = df['date'].dt.to_period('M').astype(str)

    # Weekend indicator
    df['is_weekend'] = df['dayofweek'] >= 5

    # Time period of day
    # Use unique labels to avoid pandas Categorical error when ordered=True
    df['time_period'] = pd.cut(
        df['hour'],
        bins=[-1, 6, 9, 12, 18, 21, 24],
        labels=['凌晨', '早晨', '上午', '下午', '傍晚', '夜間']
    )

    # ===== Space Dimension (S) =====
    # Region classification (support both Chinese and English county names)
    region_map = {
        # Chinese names
        '台北市': '北部', '新北市': '北部', '基隆市': '北部',
        '桃園市': '北部', '新竹市': '北部', '新竹縣': '北部',
        '台中市': '中部', '彰化縣': '中部', '南投縣': '中部',
        '苗栗縣': '中部', '雲林縣': '中部',
        '高雄市': '南部', '台南市': '南部', '屏東縣': '南部',
        '嘉義市': '南部', '嘉義縣': '南部',
        '花蓮縣': '東部', '台東縣': '東部',
        '澎湖縣': '離島', '金門縣': '離島', '連江縣': '離島',

        # English names
        'Taipei City': '北部', 'New Taipei City': '北部', 'Keelung City': '北部',
        'Taoyuan City': '北部', 'Hsinchu City': '北部', 'Hsinchu County': '北部',
        'Taichung City': '中部', 'Changhua County': '中部', 'Nantou County': '中部',
        'Miaoli County': '中部', 'Yunlin County': '中部',
        'Kaohsiung City': '南部', 'Tainan City': '南部', 'Pingtung County': '南部',
        'Chiayi City': '南部', 'Chiayi County': '南部',
        'Hualien County': '東部', 'Taitung County': '東部',
        'Penghu County': '離島', 'Kinmen County': '離島', 'Lienchiang County': '離島'
    }
    df['region'] = df['county'].map(region_map)

    # Log if any counties are not mapped
    unmapped_counties = df[df['region'].isna()]['county'].unique()
    if len(unmapped_counties) > 0:
        logger.warning(f"Unmapped counties found: {unmapped_counties}")

    # ===== Pollutant Dimension (P) =====
    # AQI level classification
    def classify_aqi(aqi):
        if pd.isna(aqi):
            return None
        elif aqi <= 50:
            return '良好'
        elif aqi <= 100:
            return '普通'
        elif aqi <= 150:
            return '對敏感族群不健康'
        elif aqi <= 200:
            return '不健康'
        elif aqi <= 300:
            return '非常不健康'
        else:
            return '危害'

    df['aqi_level'] = df['aqi'].apply(classify_aqi)

    # Pollutant category
    pollutant_category_map = {
        'PM2.5': '懸浮微粒',
        'PM10': '懸浮微粒',
        'O3': '氣態污染物',
        'CO': '氣態污染物',
        'SO2': '氣態污染物',
        'NO2': '氣態污染物',
        'NOx': '氣態污染物'
    }
    df['pollutant_category'] = df['pollutant'].map(pollutant_category_map)

    # ===== Condition Dimension (C) =====
    # Wind speed level using unified Beaufort scale classification
    # Note: Using apply() with categorize_windspeed() ensures consistency across all pages
    df['wind_level'] = df['windspeed'].apply(categorize_windspeed)

    # Pollution status indicator
    df['is_exceed'] = df['aqi'] > 100

    logger.info(f"Data prepared: {len(df)} rows with SPCT dimension labels")

    return df


def 空氣質量結構(df: pd.DataFrame, group_by: str = 'county') -> pd.DataFrame:
    """
    Calculate air quality structure metrics (analogous to sales structure).

    This function computes aggregated metrics similar to the 成交結構 function
    in the course reference code, adapted for air quality data.

    Args:
        df: Air quality DataFrame
        group_by: Column name to group by (e.g., 'county', 'yq', 'season')

    Returns:
        DataFrame with air quality structure metrics

    Metrics Calculated:
        - 監測站數 (number of stations)
        - 測量次數 (number of measurements)
        - 平均AQI, AQI中位數, 最高AQI, 最低AQI
        - 平均PM2.5, 平均PM10, 平均O3
        - 達標率 (compliance rate: AQI <= 100)

    Example:
        >>> structure = 空氣質量結構(df, 'county')
        >>> structure[['county', '平均AQI', '達標率']]
    """
    result = df.groupby(group_by).agg({
        'sitename': 'nunique',
        'date': 'count',
        'aqi': ['mean', 'median', 'max', 'min'],
        'pm2.5': 'mean',
        'pm10': 'mean',
        'o3': 'mean'
    }).reset_index()

    # Flatten multi-level columns
    result.columns = [
        group_by,
        '監測站數', '測量次數',
        '平均AQI', 'AQI中位數', '最高AQI', '最低AQI',
        '平均PM2.5', '平均PM10', '平均O3'
    ]

    # Calculate derived metrics
    # Compliance rate: percentage of measurements with AQI <= 100
    compliance = df[df['aqi'] <= 100].groupby(group_by).size()
    total = df.groupby(group_by).size()
    result['達標率'] = (compliance / total * 100).fillna(0).round(1)

    # Average measurements per station
    result['站均測量次數'] = (result['測量次數'] / result['監測站數']).round(0)

    # Round numeric columns
    result['平均AQI'] = result['平均AQI'].round(1)
    result['AQI中位數'] = result['AQI中位數'].round(1)
    result['平均PM2.5'] = result['平均PM2.5'].round(1)
    result['平均PM10'] = result['平均PM10'].round(1)
    result['平均O3'] = result['平均O3'].round(1)

    logger.info(f"Air quality structure calculated for {len(result)} groups")

    return result


def get_aqi_color(aqi: float) -> str:
    """
    Get color code for AQI level visualization.

    Args:
        aqi: AQI value

    Returns:
        Hex color code
    """
    if pd.isna(aqi):
        return "#CCCCCC"
    elif aqi <= 50:
        return "#00E400"  # Green - Good
    elif aqi <= 100:
        return "#FFFF00"  # Yellow - Moderate
    elif aqi <= 150:
        return "#FF7E00"  # Orange - Unhealthy for sensitive groups
    elif aqi <= 200:
        return "#FF0000"  # Red - Unhealthy
    elif aqi <= 300:
        return "#8F3F97"  # Purple - Very unhealthy
    else:
        return "#7E0023"  # Maroon - Hazardous


def get_aqi_recommendation(aqi: float, user_group: str = "一般民眾") -> str:
    """
    Generate health recommendation based on AQI level and user group.

    Args:
        aqi: Current AQI value
        user_group: User group category

    Returns:
        Health recommendation text in Traditional Chinese
    """
    advice_matrix = {
        '一般民眾': {
            (0, 50): "✅ 空氣品質良好，適合各種戶外活動。",
            (51, 100): "✅ 空氣品質普通，可正常戶外活動。",
            (101, 150): "⚠️ 建議減少長時間劇烈運動。",
            (151, 200): "🚫 應減少戶外活動，外出時配戴口罩。",
            (201, 500): "⛔ 避免戶外活動，關閉門窗，使用空氣清淨機。"
        },
        '敏感族群': {
            (0, 50): "✅ 空氣品質良好，可正常活動。",
            (51, 100): "⚠️ 空氣品質普通，注意身體狀況，減少劇烈活動。",
            (101, 150): "🚫 應減少戶外活動，必要外出時配戴口罩。",
            (151, 500): "⛔ 避免所有戶外活動，留在室內並使用空氣清淨機。"
        },
        '戶外工作者': {
            (0, 50): "✅ 空氣品質良好，可正常工作。",
            (51, 100): "✅ 空氣品質普通，可正常工作，多補充水分。",
            (101, 150): "⚠️ 建議縮短戶外工作時間，配戴口罩，多休息。",
            (151, 500): "🚫 應暫停戶外工作或採取防護措施，頻繁休息。"
        },
        '運動愛好者': {
            (0, 50): "✅ 空氣品質良好，適合各種運動。",
            (51, 100): "✅ 空氣品質普通，可正常運動。",
            (101, 150): "⚠️ 減少高強度運動，改為室內運動。",
            (151, 500): "🚫 避免戶外運動，建議改為室內運動或休息。"
        }
    }

    if pd.isna(aqi):
        return "數據異常，請查看最新官方公告。"

    for (min_aqi, max_aqi), advice in advice_matrix.get(user_group, advice_matrix['一般民眾']).items():
        if min_aqi <= aqi <= max_aqi:
            return advice

    return "數據異常，請查看最新官方公告。"


def filter_data(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    counties: Optional[List[str]] = None,
    stations: Optional[List[str]] = None,
    pollutants: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter DataFrame based on multiple criteria.

    Args:
        df: Input DataFrame
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        counties: List of counties to include
        stations: List of stations to include
        pollutants: List of pollutants to filter by

    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()

    if start_date:
        filtered_df = filtered_df[filtered_df['date'] >= pd.to_datetime(start_date)]

    if end_date:
        filtered_df = filtered_df[filtered_df['date'] <= pd.to_datetime(end_date)]

    if counties and len(counties) > 0:
        filtered_df = filtered_df[filtered_df['county'].isin(counties)]

    if stations and len(stations) > 0:
        filtered_df = filtered_df[filtered_df['sitename'].isin(stations)]

    if pollutants and len(pollutants) > 0:
        # This filter is for primary pollutant
        filtered_df = filtered_df[filtered_df['pollutant'].isin(pollutants)]

    logger.info(f"Filtered data: {len(filtered_df)} rows")

    return filtered_df
