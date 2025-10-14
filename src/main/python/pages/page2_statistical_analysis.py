"""
Page 2: Statistical Analysis (統計分析) - Information Layer

This page implements the "Information" level of the DIKW hierarchy, displaying:
- KPI metrics with comparisons
- Time series trends
- Crosstab analysis
- Distribution charts
- Statistical summaries

Author: Claude Code
Date: 2025-10-14
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.app_utils import 空氣質量結構, get_aqi_color
from utils.app_viz import (
    create_time_series_plot,
    create_crosstab_heatmap,
    create_distribution_plot,
    create_bar_chart,
    create_trend_with_moving_average
)


def render(df: pd.DataFrame):
    """
    Render the Statistical Analysis page.

    Args:
        df: Air quality DataFrame with SPCT dimension labels
    """
    st.header("📈 統計分析 - Information Layer")
    st.markdown("### DIKW層級：Information（資訊）")
    st.markdown("**問題：有多少？** - 經過處理、彙總、有意義的數據")
    st.markdown("---")

    # ===== KPI Metrics Section =====
    st.subheader("1️⃣ 關鍵指標 (KPI)")

    # Calculate KPIs
    avg_aqi = df['aqi'].mean()
    median_aqi = df['aqi'].median()
    max_aqi = df['aqi'].max()
    compliance_rate = (df['aqi'] <= 100).sum() / len(df) * 100
    main_pollutant = df['pollutant'].mode()[0] if len(df) > 0 else "N/A"

    # Display KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="平均AQI",
            value=f"{avg_aqi:.1f}",
            delta=f"中位數 {median_aqi:.0f}",
            help="所有記錄的平均空氣質量指數"
        )

    with col2:
        st.metric(
            label="達標率",
            value=f"{compliance_rate:.1f}%",
            delta="AQI ≤ 100",
            delta_color="normal",
            help="空氣質量良好（AQI≤100）的比例"
        )

    with col3:
        st.metric(
            label="最高AQI",
            value=f"{max_aqi:.0f}",
            help="記錄中的最高AQI值"
        )

    with col4:
        unique_stations = df['sitename'].nunique()
        st.metric(
            label="監測站數",
            value=f"{unique_stations}",
            help="涵蓋的監測站數量"
        )

    with col5:
        st.metric(
            label="主要污染物",
            value=main_pollutant,
            help="最常見的主要污染物類型"
        )

    # AQI level distribution
    st.markdown("#### AQI等級分布")
    if 'aqi_level' in df.columns:
        aqi_level_counts = df['aqi_level'].value_counts()

        col1, col2 = st.columns([2, 1])

        with col1:
            import plotly.express as px

            fig = px.pie(
                values=aqi_level_counts.values,
                names=aqi_level_counts.index,
                title='AQI等級分布'
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>數量: %{value}<br>比例: %{percent}<extra></extra>'
            )
            fig.update_layout(
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial, Microsoft YaHei, sans-serif",
                    font_color="black"
                )
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.dataframe(
                pd.DataFrame({
                    'AQI等級': aqi_level_counts.index,
                    '記錄數': aqi_level_counts.values,
                    '比例(%)': (aqi_level_counts.values / len(df) * 100).round(1)
                }),
                width='stretch'
            )

    # ===== Time Series Analysis =====
    st.markdown("---")
    st.subheader("2️⃣ 時間序列分析")

    tab1, tab2, tab3 = st.tabs(["AQI趨勢", "污染物趨勢", "移動平均"])

    with tab1:
        st.markdown("#### AQI時間序列")

        # Aggregate by date
        daily_aqi = df.groupby(df['date'].dt.date)['aqi'].mean().reset_index()
        daily_aqi.columns = ['date', 'aqi']
        daily_aqi['date'] = pd.to_datetime(daily_aqi['date'])

        fig = create_time_series_plot(daily_aqi, 'aqi', 'AQI日平均趨勢')
        st.plotly_chart(fig, width='stretch')

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均值", f"{daily_aqi['aqi'].mean():.1f}")
        with col2:
            st.metric("標準差", f"{daily_aqi['aqi'].std():.1f}")
        with col3:
            trend = "上升" if daily_aqi['aqi'].iloc[-1] > daily_aqi['aqi'].iloc[0] else "下降"
            st.metric("趨勢", trend)

    with tab2:
        st.markdown("#### 主要污染物濃度趨勢")

        pollutant_col = st.selectbox(
            "選擇污染物",
            ['pm2.5', 'pm10', 'o3', 'co', 'so2', 'no2'],
            key='pollutant_ts'
        )

        if pollutant_col in df.columns:
            daily_pollutant = df.groupby(df['date'].dt.date)[pollutant_col].mean().reset_index()
            daily_pollutant.columns = ['date', pollutant_col]
            daily_pollutant['date'] = pd.to_datetime(daily_pollutant['date'])

            fig = create_time_series_plot(
                daily_pollutant,
                pollutant_col,
                f'{pollutant_col.upper()} 日平均趨勢',
                show_thresholds=False
            )
            st.plotly_chart(fig, width='stretch')

    with tab3:
        st.markdown("#### 移動平均分析")

        window_size = st.slider("移動平均窗口（天）", 3, 30, 7)

        daily_aqi_ma = df.groupby(df['date'].dt.date)['aqi'].mean().reset_index()
        daily_aqi_ma.columns = ['date', 'aqi']
        daily_aqi_ma['date'] = pd.to_datetime(daily_aqi_ma['date'])

        fig = create_trend_with_moving_average(
            daily_aqi_ma,
            'aqi',
            window=window_size,
            title=f'AQI趨勢分析（{window_size}日移動平均）'
        )
        st.plotly_chart(fig, width='stretch')

    # ===== Crosstab Analysis =====
    st.markdown("---")
    st.subheader("3️⃣ 交叉表分析 (Crosstab)")

    tab1, tab2 = st.tabs(["縣市 × 月份", "區域 × 季節"])

    with tab1:
        st.markdown("#### 各縣市各月份平均AQI")

        if 'month' in df.columns:
            # Create pivot table
            pivot_data = df.pivot_table(
                index='county',
                columns='month',
                values='aqi',
                aggfunc='mean'
            )

            # Display heatmap
            fig = create_crosstab_heatmap(
                df,
                'month',
                'county',
                'aqi',
                'mean',
                '各縣市各月份平均AQI熱力圖'
            )
            st.plotly_chart(fig, width='stretch')

            # Display table
            with st.expander("查看詳細數據表"):
                st.dataframe(pivot_data.round(1), width='stretch')

    with tab2:
        st.markdown("#### 各區域各季節平均AQI")

        if 'region' in df.columns and 'season' in df.columns:
            fig = create_crosstab_heatmap(
                df,
                'season',
                'region',
                'aqi',
                'mean',
                '各區域各季節平均AQI熱力圖'
            )
            st.plotly_chart(fig, width='stretch')

            # Display statistics
            regional_stats = 空氣質量結構(df, 'region')
            st.dataframe(regional_stats, width='stretch')

    # ===== Distribution Analysis =====
    st.markdown("---")
    st.subheader("4️⃣ 分布分析")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### AQI分布直方圖")

        fig = create_distribution_plot(df, 'aqi', 'histogram', 'AQI分布')
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### AQI箱型圖")

        fig = create_distribution_plot(df, 'aqi', 'box', 'AQI箱型圖')
        st.plotly_chart(fig, width='stretch')

    # Pollutant distributions
    st.markdown("#### 污染物濃度分布")

    pollutant_cols = ['pm2.5', 'pm10', 'o3']
    pollutant_cols = [col for col in pollutant_cols if col in df.columns]

    if pollutant_cols:
        selected_pollutant = st.selectbox(
            "選擇污染物",
            pollutant_cols,
            key='pollutant_dist'
        )

        col1, col2 = st.columns(2)

        with col1:
            fig = create_distribution_plot(
                df,
                selected_pollutant,
                'histogram',
                f'{selected_pollutant.upper()} 分布'
            )
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig = create_distribution_plot(
                df,
                selected_pollutant,
                'box',
                f'{selected_pollutant.upper()} 箱型圖'
            )
            st.plotly_chart(fig, width='stretch')

    # ===== Air Quality Structure Analysis =====
    st.markdown("---")
    st.subheader("5️⃣ 空氣質量結構分析")

    tab1, tab2, tab3 = st.tabs(["按縣市", "按季節", "按年度"])

    with tab1:
        st.markdown("#### 各縣市空氣質量結構")

        county_structure = 空氣質量結構(df, 'county')
        county_structure = county_structure.sort_values('平均AQI', ascending=False)

        st.dataframe(county_structure, width='stretch')

        # Visualize
        fig = create_bar_chart(
            county_structure.head(15),
            'county',
            '平均AQI',
            orientation='h',
            title='各縣市平均AQI (前15名)'
        )
        st.plotly_chart(fig, width='stretch')

    with tab2:
        if 'season' in df.columns:
            st.markdown("#### 各季節空氣質量結構")

            season_structure = 空氣質量結構(df, 'season')

            st.dataframe(season_structure, width='stretch')

            fig = create_bar_chart(
                season_structure,
                'season',
                '平均AQI',
                title='各季節平均AQI'
            )
            st.plotly_chart(fig, width='stretch')

    with tab3:
        if 'year' in df.columns:
            st.markdown("#### 各年度空氣質量結構")

            year_structure = 空氣質量結構(df, 'year')
            year_structure = year_structure.sort_values('year')

            st.dataframe(year_structure, width='stretch')

            fig = create_bar_chart(
                year_structure,
                'year',
                '平均AQI',
                title='各年度平均AQI趨勢'
            )
            st.plotly_chart(fig, width='stretch')

    # ===== Summary =====
    st.markdown("---")

    # Calculate key insights
    best_county = county_structure.iloc[-1]['county']
    worst_county = county_structure.iloc[0]['county']
    best_aqi = county_structure.iloc[-1]['平均AQI']
    worst_aqi = county_structure.iloc[0]['平均AQI']

    st.success(f"""
    ### 📊 統計分析摘要

    - **整體表現**: 平均AQI為 {avg_aqi:.1f}，達標率為 {compliance_rate:.1f}%
    - **空氣最佳**: {best_county} (平均AQI: {best_aqi:.1f})
    - **需要改善**: {worst_county} (平均AQI: {worst_aqi:.1f})
    - **主要污染物**: {main_pollutant}

    💡 **下一步建議**:
    - 前往「規律發現」頁面深入分析這些統計結果背後的規律
    - 在「智慧決策」頁面獲取改善建議
    """)
