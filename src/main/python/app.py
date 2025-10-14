"""
Taiwan Air Quality Data Analysis System - Main Application

This is the main Streamlit application implementing the DIKW hierarchy
(Data → Information → Knowledge → Wisdom) for air quality analysis.

Based on:
- PLANNING.md: System architecture and SPCT model
- AIp04空間與網站X.py: Navigation and session state patterns

Features:
- 5 pages corresponding to DIKW layers
- Interactive sidebar controls for data filtering
- Session state management for data persistence
- Operation logging system

Usage:
    streamlit run src/main/python/app.py

Author: Claude Code
Date: 2025-10-14
"""

import streamlit as st
from streamlit_navigation_bar import st_navbar
import sys
from pathlib import Path
import os
import logging

# Add current directory to Python path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from utils.app_utils import init_session_state, add_log, prepare_data, filter_data
from utils.data_loader import AirQualityDataLoader
import pandas as pd
from datetime import datetime, timedelta


# ==================== Page Configuration ====================

st.set_page_config(
    page_title="台灣空氣質量分析系統",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure the main view container remains scrollable in case any component
# injects restrictive CSS (observed in some nav bar themes)
st.markdown(
    """
    <style>
    html, body, #root, .stApp {
        height: 100% !important;
        overflow-y: auto !important;
    }
    [data-testid="stAppViewContainer"] {
        overflow-y: auto !important;
        overflow-x: hidden;
    }
    main.block-container, section.main {
        overflow: visible !important;
    }
    [data-testid="stHeader"], header {
        position: sticky; top: 0; z-index: 10;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================== Navigation Bar ====================

pages = [
    "[數據總覽]",      # Data Layer
    "[統計分析]",      # Information Layer
    "[規律發現]",      # Knowledge Layer
    "[智慧決策]",      # Wisdom Layer
    "[預測模型]"       # Wisdom Layer Advanced
]

use_top_nav = os.getenv("USE_TOP_NAV", "0").lower() in ("1", "true", "yes", "y")
if use_top_nav:
    page = st_navbar(pages)
    nav_mode = "navbar"
else:
    page = st.sidebar.radio("頁面導航", pages, index=0)
    nav_mode = "sidebar"

# Re-apply scroll CSS after navbar renders to ensure our overrides win
st.markdown(
    """
    <style>
    /* post-navbar scroll fix */
    html, body, #root, .stApp { height: 100% !important; overflow-y: auto !important; }
    [data-testid=\"stAppViewContainer\"] { overflow-y: auto !important; overflow-x: hidden; }
    main.block-container, section.main { overflow: visible !important; }
    [data-testid=\"stHeader\"], header { position: sticky; top: 0; z-index: 10; }
    </style>
    """,
    unsafe_allow_html=True
)

# Debug UI toggle and logger
DEBUG_UI = os.getenv("DEBUG_UI", "0").lower() in ("1", "true", "yes", "y")
CSS_PATCH_VERSION = "scrollfix-2025-10-14#post"
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
if DEBUG_UI:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🐛 Debug Info")
    st.sidebar.text(f"Nav mode: {nav_mode}")
    st.sidebar.text(f"CSS patch: {CSS_PATCH_VERSION}")
    st.sidebar.text(f"Page: {page}")
    logger.info(f"[DEBUG] Nav mode={nav_mode}, CSS={CSS_PATCH_VERSION}, Page={page}")

## Remove duplicate/legacy debug block (migrated above)


# ==================== Session State Initialization ====================

sss = init_session_state("台灣空氣質量分析系統")


# ==================== Main Title ====================

st.title("🌍 台灣空氣質量分析系統")
st.markdown("### 基於DIKW原則的多維度空氣質量分析平台")
st.markdown("---")


# ==================== Sidebar Control Panel ====================

st.sidebar.title("📊 空氣質量分析控制盤")
st.sidebar.markdown("---")


# ===== Data Loading Section =====
st.sidebar.header("🔧 【數據載入】")

@st.cache_resource
def get_data_loader():
    """Get singleton data loader instance"""
    return AirQualityDataLoader()


@st.cache_data
def load_initial_data():
    """Load initial data for date range and county options"""
    try:
        loader = get_data_loader()
        min_date, max_date = loader.get_date_range()
        # Derive stations directly from the fact table to keep naming consistent
        station_source = "air_quality"
        try:
            stations = loader.query_db(
                "SELECT DISTINCT sitename, county FROM air_quality ORDER BY sitename"
            )
        except Exception:
            stations = loader.get_station_list()
            station_source = "station_metadata/fallback"
        return min_date, max_date, stations
    except Exception as e:
        st.sidebar.error(f"無法載入數據元信息: {e}")
        return None, None, None


# Load metadata
min_date, max_date, stations_df = load_initial_data()

if min_date is None:
    st.error("⚠️ 無法連接到數據源。請確認數據文件存在。")
    st.info("""
    請確認以下文件存在：
    - `data/processed/air_quality.duckdb` 或
    - `data/processed/*.parquet`
    """)
    st.stop()

st.sidebar.success(f"✅ 數據可用期間: {min_date.date()} ~ {max_date.date()}")
if DEBUG_UI:
    try:
        # Show station source if available in closure (best-effort; harmless if not)
        st.sidebar.caption("[DEBUG] Station list source: air_quality (distinct)")
    except Exception:
        pass


# ===== Data Filtering Section =====
st.sidebar.header("🔍 【數據篩選】")

# Date range selection
default_end_date = max_date.date()
default_start_date = (max_date - timedelta(days=30)).date()

date_range = st.sidebar.date_input(
    "選擇時間範圍",
    value=(default_start_date, default_end_date),
    min_value=min_date.date(),
    max_value=max_date.date(),
    help="選擇要分析的日期區間"
)

# Handle date range selection
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = date_range
    end_date = date_range
# Persist active date range for pages
st.session_state['date_range'] = (start_date, end_date)

# County selection
if stations_df is not None:
    available_counties = sorted(stations_df['county'].unique())

    # Create safe default list - only use counties that actually exist
    preferred_defaults = ['台北市', '新北市', '台中市', '高雄市']
    safe_defaults = [c for c in preferred_defaults if c in available_counties]

    # If no preferred counties found, use first 4 available counties
    if not safe_defaults and len(available_counties) >= 4:
        safe_defaults = list(available_counties[:4])
    elif not safe_defaults and len(available_counties) > 0:
        safe_defaults = [available_counties[0]]
    else:
        safe_defaults = []

    selected_counties = st.sidebar.multiselect(
        "選擇縣市",
        options=available_counties,
        default=safe_defaults,
        help="選擇要分析的縣市（可多選）"
    )
    # Persist counties selection for pages
    st.session_state['selected_counties'] = selected_counties

    # Station selection (filtered by selected counties AND date range for accuracy)
    try:
        loader_for_station = get_data_loader()
        base_where = f"date BETWEEN '{default_start_date}' AND '{default_end_date}'"
        # Use current sidebar date selection for availability
        base_where = f"date BETWEEN '{start_date}' AND '{end_date}'"
        if selected_counties:
            counties_str = "', '".join([c.replace("'", "''") for c in selected_counties])
            sql_st = f"""
                SELECT DISTINCT sitename
                FROM air_quality
                WHERE {base_where} AND county IN ('{counties_str}')
                ORDER BY sitename
            """
        else:
            sql_st = f"""
                SELECT DISTINCT sitename
                FROM air_quality
                WHERE {base_where}
                ORDER BY sitename
            """
        st_df = loader_for_station.query_db(sql_st)
        available_stations = st_df['sitename'].tolist()
    except Exception:
        # Fallback to preloaded station list if DB query fails
        if selected_counties:
            available_stations = sorted(
                stations_df[stations_df['county'].isin(selected_counties)]['sitename'].unique()
            )
        else:
            available_stations = sorted(stations_df['sitename'].unique())

    # Preserve prior selection but drop any stations not available in current window
    prev_selected = list(st.session_state.get('selected_stations', []))
    default_selected = [s for s in prev_selected if s in available_stations]
    if prev_selected and len(default_selected) < len(prev_selected):
        st.sidebar.warning("部分先前選擇的站點在目前時間/縣市條件下無資料，已自動移除。")
    selected_stations = st.sidebar.multiselect(
        "選擇監測站",
        options=available_stations,
        default=default_selected,
        key='selected_stations',
        help="依目前選擇的縣市與時間範圍顯示可用站點。留空＝全部"
    )

    # Compute whether loading makes sense (avoid empty result when counties selected but無站點)
    can_load = True
    if selected_counties and len(available_stations) == 0:
        can_load = False
        st.sidebar.error("當前時間範圍內，所選縣市沒有任何有效監測站。請調整條件。")
else:
    selected_counties = []
    selected_stations = []

# Pollutant selection
pollutant_options = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
selected_pollutants = st.sidebar.multiselect(
    "主要污染物篩選",
    options=pollutant_options,
    default=['PM2.5', 'PM10'],
    help="選擇要關注的污染物類型"
)


# ===== Analysis Parameters Section =====
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 【分析參數】")

stat_method = st.sidebar.selectbox(
    "統計方法",
    ["平均值", "中位數", "最大值", "最小值"],
    help="選擇數據聚合方式"
)

time_agg = st.sidebar.radio(
    "時間聚合層級",
    ["小時", "日", "週", "月", "季", "年"],
    index=1,
    help="選擇時間維度的聚合單位"
)

aqi_threshold = st.sidebar.slider(
    "AQI警示閾值",
    min_value=0,
    max_value=200,
    value=100,
    step=10,
    help="設定空氣品質警示的AQI閾值"
)


# ===== Load Data Button =====
st.sidebar.markdown("---")

if st.sidebar.button("🔄 載入/更新數據", type="primary", disabled=not can_load):
    with st.spinner("載入數據中..."):
        try:
            loader = get_data_loader()

            # Build SQL query for efficient loading
            def _escape(values):
                return [v.replace("'", "''") for v in values]

            base_where = f"date BETWEEN '{start_date}' AND '{end_date}'"

            if selected_stations:
                # Prefer station filter to avoid county label mismatches excluding stations
                stations_esc = "', '".join(_escape(selected_stations))
                sql = f"""
                    SELECT *
                    FROM air_quality
                    WHERE {base_where}
                      AND sitename IN ('{stations_esc}')
                """
            elif selected_counties:
                counties_esc = "', '".join(_escape(selected_counties))
                sql = f"""
                    SELECT *
                    FROM air_quality
                    WHERE {base_where}
                      AND county IN ('{counties_esc}')
                """
            else:
                sql = f"""
                    SELECT *
                    FROM air_quality
                    WHERE {base_where}
                """

            if DEBUG_UI:
                st.sidebar.markdown("#### [DEBUG] SQL")
                st.sidebar.code(sql.strip())

            # Load data
            df = loader.query_db(sql)

            # Save available stations before filtering for error messages
            available_stations_in_data = df['sitename'].unique()

            # Debug: Show actual data format
            st.sidebar.info(f"""
            🔍 Debug 資訊：
            - SQL查詢後記錄數: {len(df):,}
            - 唯一縣市: {df['county'].nunique()} 個
            - 縣市範例: {', '.join(df['county'].unique()[:3])}
            - 唯一站點: {df['sitename'].nunique()} 個
            - 站點範例: {', '.join(available_stations_in_data[:3])}
            """)

            # Apply station filter if specified
            if selected_stations:
                st.sidebar.markdown("### 🔍 站點匹配詳細診斷")

                # Show user selections with detailed format
                st.sidebar.write("**用戶選擇的站點：**")
                for i, station in enumerate(selected_stations, 1):
                    st.sidebar.text(f"{i}. '{station}' (長度: {len(station)})")

                # Show available stations in data with detailed format
                st.sidebar.write("**資料庫中符合條件的站點：**")
                for i, station in enumerate(available_stations_in_data[:10], 1):
                    st.sidebar.text(f"{i}. '{station}' (長度: {len(station)})")

                # Perform matching and show results
                st.sidebar.write("**匹配結果：**")
                matched = []
                unmatched = []
                for station in selected_stations:
                    if station in available_stations_in_data:
                        matched.append(station)
                        st.sidebar.success(f"✅ '{station}' - 匹配成功")
                    else:
                        unmatched.append(station)
                        st.sidebar.error(f"❌ '{station}' - 未找到")
                        # Try to find similar names
                        similar = [s for s in available_stations_in_data if station.lower() in s.lower() or s.lower() in station.lower()]
                        if similar:
                            st.sidebar.info(f"   💡 相似名稱: {similar[:3]}")

                # Sanity-check in database: exact-match existence per station (all time and current range)
                st.sidebar.write("**資料庫精確檢查：**")
                for station in selected_stations:
                    s_esc = station.replace("'", "''")
                    try:
                        exists_df = loader.query_db(
                            f"SELECT COUNT(*) AS cnt FROM air_quality WHERE sitename = '{s_esc}'"
                        )
                        cnt = int(exists_df['cnt'].iloc[0]) if not exists_df.empty else 0
                        range_df = loader.query_db(
                            f"SELECT COUNT(*) AS cnt FROM air_quality WHERE sitename = '{s_esc}' AND date BETWEEN '{start_date}' AND '{end_date}'"
                        )
                        cnt_range = int(range_df['cnt'].iloc[0]) if not range_df.empty else 0
                    except Exception:
                        cnt = -1
                        cnt_range = -1
                    st.sidebar.text(f"- '{station}': 全期間 {cnt} 筆 | 目前區間 {cnt_range} 筆")

                # Prefix LIKE check (before parenthesis) to show similar entries in fact table
                st.sidebar.write("**資料庫近似檢查（前綴 LIKE）：**")
                for station in selected_stations:
                    prefix = station.split(' (')[0].strip()
                    p_esc = prefix.replace("'", "''")
                    try:
                        like_df = loader.query_db(
                            f"""
                            SELECT DISTINCT sitename
                            FROM air_quality
                            WHERE sitename ILIKE '{p_esc}%'
                            ORDER BY sitename
                            """
                        )
                        examples = like_df['sitename'].tolist()[:5]
                    except Exception:
                        examples = []
                    st.sidebar.text(f"- 前綴 '{prefix}': {examples}")

                # Apply filter
                df_before = len(df)
                stations_before = df['sitename'].nunique()
                df = df[df['sitename'].isin(selected_stations)]
                stations_after = df['sitename'].nunique()

                st.sidebar.info(f"🔧 站點過濾: {df_before:,} → {len(df):,} 筆記錄")
                st.sidebar.info(f"🔧 過濾後唯一站點數: {stations_after} / {len(selected_stations)}")

                # Validation: Check if filtering worked
                if len(df) == 0:
                    st.sidebar.error("❌ 站點過濾後無數據！所有選擇的站點都未匹配。")
                    st.stop()  # Stop execution if no data
                elif stations_after < len(selected_stations):
                    st.sidebar.warning(f"⚠️ 只有 {stations_after}/{len(selected_stations)} 個站點成功載入")
                    st.sidebar.info(f"成功載入: {list(df['sitename'].unique())}")
                    st.sidebar.info(f"未能載入: {unmatched}")

            # Prepare data with SPCT dimension labels
            df = prepare_data(df)

            # Debug: Show region mapping result
            if 'region' in df.columns:
                nan_count = df['region'].isna().sum()
                if nan_count > 0:
                    st.sidebar.warning(f"⚠️ 區域映射失敗: {nan_count} 筆記錄的區域為 NaN")
                    st.sidebar.info(f"未映射的縣市: {df[df['region'].isna()]['county'].unique()}")

            # Store in session state
            sss.df = df
            sss.df_filtered = df

            # Log the operation
            add_log(f"載入數據: {len(df):,} 筆記錄 ({start_date} ~ {end_date})")

            st.sidebar.success(f"✅ 成功載入 {len(df):,} 筆記錄，{df['sitename'].nunique()} 個站點")

        except Exception as e:
            st.sidebar.error(f"❌ 載入數據失敗: {e}")
            add_log(f"數據載入失敗: {e}")


# ===== Current Status Section =====
st.sidebar.markdown("---")
st.sidebar.header("📈 【當前狀態】")

if sss.df is not None:
    st.sidebar.metric("數據筆數", f"{len(sss.df):,}")
    st.sidebar.metric("監測站數", f"{sss.df['sitename'].nunique()}")
    st.sidebar.metric("時間範圍", f"{(sss.df['date'].max() - sss.df['date'].min()).days} 天")

    if len(sss.df) > 0:
        avg_aqi = sss.df['aqi'].mean()
        st.sidebar.metric("平均AQI", f"{avg_aqi:.1f}")
else:
    st.sidebar.info("尚未載入數據")


# User input section removed per requirement (no user feature)


# ===== Operation Log Section =====
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 操作LOG日誌")

# Display last 10 log entries
for i, log_entry in enumerate(sss.log[-10:], 1):
    st.sidebar.text(f"{len(sss.log) - 10 + i}. {log_entry}")


# ==================== Page Routing ====================

# Check if data is loaded
if sss.df is None:
    st.warning("⚠️ 請先在側邊欄載入數據以開始分析")
    st.info("""
    ### 使用說明（側邊欄控制盤）

    控制盤區塊說明：
    - 【數據載入】顯示可用日期範圍；若資料不存在會提示檢查路徑
    - 【數據篩選】依序選擇：
      - 時間範圍：開始與結束日期（必選）
      - 縣市：可多選；留空表示全部
      - 監測站：會依選擇的縣市篩出站點；留空表示全部
      - 主要污染物：僅篩選主污染物欄位
    - 【分析參數】
      - 統計方法：平均值/中位數/最大值/最小值
      - 時間聚合層級：小時/日/週/月/季/年（部分圖表使用）
      - AQI 警示閾值：用於趨勢線與提醒
    - 「🔄 載入/更新數據」：依上述條件載入資料並套用
    - 【當前狀態】：顯示筆數、站數、資料時間範圍與平均 AQI
    - 【操作LOG日誌】：顯示最近操作與提示

    使用步驟：
    1. 在【數據篩選】選好時間、縣市（可選）、站點（可選）
    2. 視需要調整【分析參數】
    3. 按「🔄 載入/更新數據」進行載入
    4. 透過上方導航列切換各分析頁面

    小提醒：
    - 縣市/站點留空 = 不限制（全選）
    - 重新調整條件後，需再次按下「載入/更新數據」才會生效

    ### DIKW 分析層級
    - **數據總覽**：查看原始數據與基本指標
    - **統計分析**：比較分布、統計量與趨勢
    - **規律發現**：探索相關性、熱力圖、風玫瑰圖等
    - **智慧決策**：取得健康建議與策略參考
    - **預測模型**：展示預測結果（頁面架構已備妥）
    """)
    st.stop()


# Import and render appropriate page
try:
    match page:
        case "[數據總覽]":
            from pages import page1_data_overview
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 數據總覽 (start)")
                logger.info("[DEBUG] Rendering page: 數據總覽 (start)")
            page1_data_overview.render(sss.df)
            add_log("瀏覽：數據總覽頁面")
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 數據總覽 (done)")
                logger.info("[DEBUG] Rendering page: 數據總覽 (done)")

        case "[統計分析]":
            from pages import page2_statistical_analysis
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 統計分析 (start)")
                logger.info("[DEBUG] Rendering page: 統計分析 (start)")
            page2_statistical_analysis.render(sss.df)
            add_log("瀏覽：統計分析頁面")
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 統計分析 (done)")
                logger.info("[DEBUG] Rendering page: 統計分析 (done)")

        case "[規律發現]":
            from pages import page3_pattern_discovery
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 規律發現 (start)")
                logger.info("[DEBUG] Rendering page: 規律發現 (start)")
            page3_pattern_discovery.render(sss.df)
            add_log("瀏覽：規律發現頁面")
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 規律發現 (done)")
                logger.info("[DEBUG] Rendering page: 規律發現 (done)")

        case "[智慧決策]":
            from pages import page4_wisdom_decision
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 智慧決策 (start)")
                logger.info("[DEBUG] Rendering page: 智慧決策 (start)")
            page4_wisdom_decision.render(sss.df)
            add_log("瀏覽：智慧決策頁面")
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 智慧決策 (done)")
                logger.info("[DEBUG] Rendering page: 智慧決策 (done)")

        case "[預測模型]":
            from pages import page5_prediction_model
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 預測模型 (start)")
                logger.info("[DEBUG] Rendering page: 預測模型 (start)")
            page5_prediction_model.render(sss.df)
            add_log("瀏覽：預測模型頁面")
            if DEBUG_UI:
                st.sidebar.text("[DEBUG] Rendering: 預測模型 (done)")
                logger.info("[DEBUG] Rendering page: 預測模型 (done)")

except Exception as e:
    st.error(f"❌ 頁面載入錯誤: {e}")
    st.exception(e)


# ==================== Footer ====================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>台灣空氣質量分析系統 | 基於DIKW原則與SPCT多維度模型</p>
    <p>數據來源: 台灣環保署空氣質量監測網 | 開發: Claude Code</p>
</div>
""", unsafe_allow_html=True)
