"""
Page 1: Data Overview (數據總覽) - Data Layer

This page implements the "Data" level of the DIKW hierarchy, displaying:
- Raw data table
- Data quality information
- Basic statistics
- Data structure information

Author: Claude Code
Date: 2025-10-14
"""

import streamlit as st
import pandas as pd
import numpy as np


def render(df: pd.DataFrame):
    """
    Render the Data Overview page.

    Args:
        df: Air quality DataFrame with SPCT dimension labels
    """
    st.header("📊 數據總覽 - Data Layer")
    st.markdown("### DIKW層級：Data（原始數據）")
    st.markdown("**問題：這是什麼？** - 查看原始測量值和觀察結果")
    st.markdown("---")

    # ===== Data Summary Section =====
    st.subheader("1️⃣ 數據摘要")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            label="總記錄數",
            value=f"{len(df):,}"
        )

    with col2:
        st.metric(
            label="監測站數",
            value=f"{df['sitename'].nunique()}"
        )

    with col3:
        st.metric(
            label="涵蓋縣市",
            value=f"{df['county'].nunique()}"
        )

    with col4:
        date_range_days = (df['date'].max() - df['date'].min()).days
        st.metric(
            label="時間跨度",
            value=f"{date_range_days} 天"
        )

    with col5:
        st.metric(
            label="數據欄位",
            value=f"{len(df.columns)}"
        )

    # Date range info
    st.info(f"📅 數據時間範圍: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # ===== Selection Summary & Consistency Check =====
    st.markdown("---")
    st.subheader("選擇摘要與條件對齊")

    # Read current user selections from session_state (best-effort)
    sss = st.session_state
    sel_counties = list(sss.get('selected_counties', []))
    sel_stations = list(sss.get('selected_stations', []))
    sel_date = sss.get('date_range', None)

    present_counties = sorted(df['county'].dropna().unique().tolist()) if 'county' in df.columns else []
    present_stations = sorted(df['sitename'].dropna().unique().tolist()) if 'sitename' in df.columns else []

    missing_counties = sorted(set(sel_counties) - set(present_counties)) if sel_counties else []
    missing_stations = sorted(set(sel_stations) - set(present_stations)) if sel_stations else []

    cols = st.columns(3)
    with cols[0]:
        if sel_date:
            st.markdown(f"**日期區間**: {sel_date[0]} → {sel_date[1]}")
        else:
            st.markdown("**日期區間**: 依當前資料")
    with cols[1]:
        st.markdown(f"**已選縣市**: {', '.join(sel_counties) if sel_counties else '（未限制）'}")
        if missing_counties:
            st.warning(f"以下縣市在此區間內無資料: {', '.join(missing_counties)}")
    with cols[2]:
        st.markdown(f"**已選站點**: {', '.join(sel_stations) if sel_stations else '（未限制）'}")
        if missing_stations:
            st.warning(f"以下站點在此區間內無資料: {', '.join(missing_stations)}")

    # ===== Raw Data Display =====
    st.markdown("---")
    st.subheader("2️⃣ 原始數據查看")

    # Display options
    col1, col2 = st.columns([3, 1])

    with col1:
        num_rows = st.select_slider(
            "顯示記錄數",
            options=[10, 50, 100, 500, 1000],
            value=100
        )

    with col2:
        sort_by = st.selectbox(
            "排序方式",
            ["最新記錄", "最舊記錄", "AQI最高", "AQI最低"]
        )

    # Apply sorting
    if sort_by == "最新記錄":
        df_display = df.sort_values('date', ascending=False).head(num_rows)
    elif sort_by == "最舊記錄":
        df_display = df.sort_values('date', ascending=True).head(num_rows)
    elif sort_by == "AQI最高":
        df_display = df.sort_values('aqi', ascending=False).head(num_rows)
    else:  # AQI最低
        df_display = df.sort_values('aqi', ascending=True).head(num_rows)

    # Select columns to display
    display_columns = [
        'date', 'sitename', 'county', 'aqi', 'status',
        'pm2.5', 'pm10', 'o3', 'pollutant',
        'windspeed', 'winddirec'
    ]

    # Filter to existing columns
    display_columns = [col for col in display_columns if col in df.columns]

    st.dataframe(
        df_display[display_columns],
        width='stretch',
        height=400
    )

    # Download button for filtered data
    csv = df_display.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下載當前數據為CSV",
        data=csv,
        file_name=f"air_quality_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # ===== Data Quality Check =====
    st.markdown("---")
    st.subheader("3️⃣ 數據品質檢查")

    tab1, tab2, tab3 = st.tabs(["缺失值分析", "數據類型", "基本統計"])

    with tab1:
        st.markdown("#### 缺失值統計")

        # Calculate missing values
        missing_stats = pd.DataFrame({
            '缺失數': df.isnull().sum(),
            '缺失比例(%)': (df.isnull().sum() / len(df) * 100).round(2)
        })

        # Filter to columns with missing values
        missing_stats = missing_stats[missing_stats['缺失數'] > 0]

        if len(missing_stats) > 0:
            st.warning(f"⚠️ 共有 {len(missing_stats)} 個欄位包含缺失值")
            st.dataframe(missing_stats.sort_values('缺失比例(%)', ascending=False))

            # Visualize missing data
            import plotly.express as px

            fig = px.bar(
                missing_stats.reset_index(),
                x='index',
                y='缺失比例(%)',
                title='各欄位缺失比例',
                labels={'index': '欄位名稱', '缺失比例(%)': '缺失比例 (%)'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')

        else:
            st.success("✅ 所有欄位都沒有缺失值")

    with tab2:
        st.markdown("#### 數據類型信息")

        # Get data types
        dtypes_df = pd.DataFrame({
            '欄位名稱': df.columns,
            '數據類型': df.dtypes.values,
            '非空數量': df.count().values,
            '唯一值數量': [df[col].nunique() for col in df.columns]
        })
        # Ensure Arrow-compatible types for display: cast dtype objects to string
        dtypes_df['數據類型'] = dtypes_df['數據類型'].astype(str)

        st.dataframe(dtypes_df, width='stretch')

    with tab3:
        st.markdown("#### 數值欄位基本統計")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            stats_df = df[numeric_cols].describe().T
            stats_df = stats_df.round(2)

            st.dataframe(stats_df, width='stretch')
        else:
            st.info("無數值型欄位")

    # ===== Data Structure Information =====
    st.markdown("---")
    st.subheader("4️⃣ 數據結構信息")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📍 監測站分布")

        station_count = df.groupby('county')['sitename'].nunique().sort_values(ascending=False)

        import plotly.express as px

        fig = px.bar(
            x=station_count.values,
            y=station_count.index,
            orientation='h',
            title='各縣市監測站數量',
            labels={'x': '監測站數', 'y': '縣市'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### 📈 記錄數分布")

        record_count = df.groupby('county').size().sort_values(ascending=False)

        fig = px.bar(
            x=record_count.values,
            y=record_count.index,
            orientation='h',
            title='各縣市記錄數量',
            labels={'x': '記錄數', 'y': '縣市'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # ===== Additional Information =====
    st.markdown("---")
    st.subheader("5️⃣ 維度標籤信息")

    st.markdown("""
    此數據集已經過SPCT多維度處理，包含以下維度標籤：

    - **S (空間維度)**: county, sitename, region, latitude, longitude
    - **P (污染物維度)**: pollutant, aqi, pm2.5, pm10, o3, co, so2, no2, pollutant_category, aqi_level
    - **C (條件維度)**: windspeed, winddirec, wind_level, time_period, is_weekend, is_exceed
    - **T (時間維度)**: date, year, month, day, hour, dayofweek, quarter, season, yq, ym
    """)

    # Show example of dimension labels (user-controlled sample)
    with st.expander("查看維度標籤示例"):
        if 'region' in df.columns:
            present_counties = sorted(df['county'].dropna().unique().tolist())
            # Choose default county: first in intersection with user selection, else first present
            if sel_counties:
                defaults = [c for c in present_counties if c in sel_counties]
                default_county = defaults[0] if defaults else (present_counties[0] if present_counties else None)
            else:
                default_county = present_counties[0] if present_counties else None

            if default_county is None:
                st.info("當前資料無可用縣市。")
            else:
                chosen_county = st.selectbox("範例縣市", present_counties, index=present_counties.index(default_county))
                county_df = df[df['county'] == chosen_county]
                stations_in_county = sorted(county_df['sitename'].dropna().unique().tolist())

                # Choose default station: intersection with user selection, else first
                if sel_stations:
                    s_defaults = [s for s in stations_in_county if s in sel_stations]
                    default_station = s_defaults[0] if s_defaults else (stations_in_county[0] if stations_in_county else None)
                else:
                    default_station = stations_in_county[0] if stations_in_county else None

                if default_station is None:
                    st.info("此縣市在當前區間內沒有站點資料。")
                else:
                    chosen_station = st.selectbox("範例站點", stations_in_county, index=stations_in_county.index(default_station))
                    sample = county_df[county_df['sitename'] == chosen_station]
                    if sample.empty:
                        st.warning("無法取得範例記錄。")
                    else:
                        sample_row = sample.iloc[0]
                        st.markdown("#### 範例記錄的維度標籤：")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.markdown("**空間 (S)**")
                            st.write(f"- 縣市: {sample_row.get('county', 'N/A')}")
                            st.write(f"- 區域: {sample_row.get('region', 'N/A')}")
                            st.write(f"- 站點: {sample_row.get('sitename', 'N/A')}")

                        with col2:
                            st.markdown("**污染物 (P)**")
                            st.write(f"- AQI: {sample_row.get('aqi', 'N/A')}")
                            st.write(f"- AQI等級: {sample_row.get('aqi_level', 'N/A')}")
                            st.write(f"- 主要污染物: {sample_row.get('pollutant', 'N/A')}")

                        with col3:
                            st.markdown("**條件 (C)**")
                            st.write(f"- 風速等級: {sample_row.get('wind_level', 'N/A')}")
                            st.write(f"- 時段: {sample_row.get('time_period', 'N/A')}")
                            st.write(f"- 週末: {sample_row.get('is_weekend', 'N/A')}")

                        with col4:
                            st.markdown("**時間 (T)**")
                            st.write(f"- 年: {sample_row.get('year', 'N/A')}")
                            st.write(f"- 季節: {sample_row.get('season', 'N/A')}")
                            st.write(f"- 年季: {sample_row.get('yq', 'N/A')}")

