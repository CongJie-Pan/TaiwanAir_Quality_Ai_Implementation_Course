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
import plotly.express as px
import plotly.graph_objects as go
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
    create_trend_with_moving_average,
    create_scatter_plot
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

    # ===== Bivariate Analysis =====
    st.markdown("---")
    st.subheader("2️⃣ 二維分析 (Bivariate Analysis)")
    st.markdown("**探索兩個變量之間的關係與相關性（去時間化）**")

    tab1, tab2, tab3, tab4 = st.tabs(["AQI vs 風速", "按季節分組", "其他污染物對比", "表格檢視"])

    with tab1:
        st.markdown("#### AQI 與風速的關係分析")
        st.info("💡 每個散點代表一天的平均數據，不顯示具體日期，專注於數值關係")

        # Aggregate by date (daily average)
        if 'windspeed' in df.columns and 'aqi' in df.columns:
            daily_data = df.groupby(df['date'].dt.date).agg({
                'windspeed': 'mean',
                'aqi': 'mean'
            }).reset_index()
            daily_data.columns = ['date', 'windspeed', 'aqi']

            # Remove rows with missing values
            scatter_data = daily_data[['windspeed', 'aqi']].dropna()

            if len(scatter_data) > 0:
                # Calculate correlation
                from scipy.stats import pearsonr
                corr, pvalue = pearsonr(scatter_data['windspeed'], scatter_data['aqi'])

                # Calculate linear regression slope
                import numpy as np
                z = np.polyfit(scatter_data['windspeed'], scatter_data['aqi'], 1)
                slope = z[0]  # AQI change per 1 m/s windspeed increase

                # Display statistics
                st.metric(
                    "回歸斜率",
                    f"{slope:.2f}",
                    delta="AQI per m/s",
                    help="風速每增加1 m/s，AQI平均變化量"
                )

                # Create scatter plot
                fig = create_scatter_plot(
                    df=scatter_data,
                    x_col='windspeed',
                    y_col='aqi',
                    title='AQI 與風速的關係（每日平均數據）',
                    show_trendline=True
                )
                st.plotly_chart(fig, use_container_width=True)

                # Crosstab table: Wind Speed Level × Air Quality
                st.markdown("---")
                st.markdown("#### 📋 風級 × 空氣品質交叉統計表")
                st.info("💡 表格顯示不同風級與空氣品質組合下的出現次數（天數）")

                # 定義風級分類函數
                def categorize_windspeed(ws):
                    if ws <= 1.5:
                        return '無風(0-1.5)'
                    elif ws <= 3.3:
                        return '輕風(1.6-3.3)'
                    elif ws <= 5.4:
                        return '微風(3.4-5.4)'
                    elif ws <= 7.9:
                        return '和風(5.5-7.9)'
                    else:
                        return '強風(≥8.0)'

                # 定義AQI空氣品質分類函數
                def categorize_aqi(aqi):
                    if aqi <= 50:
                        return '良好(0-50)'
                    elif aqi <= 100:
                        return '普通(51-100)'
                    elif aqi <= 150:
                        return '對敏感族群不健康(101-150)'
                    elif aqi <= 200:
                        return '不健康(151-200)'
                    elif aqi <= 300:
                        return '非常不健康(201-300)'
                    else:
                        return '危害(>300)'

                # 準備交叉表數據
                crosstab_data = scatter_data.copy()
                crosstab_data['風級'] = crosstab_data['windspeed'].apply(categorize_windspeed)
                crosstab_data['空氣品質'] = crosstab_data['aqi'].apply(categorize_aqi)

                # 創建交叉統計表（顯示出現次數）
                crosstab = pd.pivot_table(
                    crosstab_data,
                    values='aqi',
                    index='風級',
                    columns='空氣品質',
                    aggfunc='count',
                    fill_value=0
                )

                # 定義風級順序（從無風到強風）
                wind_order = ['無風(0-1.5)', '輕風(1.6-3.3)', '微風(3.4-5.4)', '和風(5.5-7.9)', '強風(≥8.0)']
                crosstab = crosstab.reindex(wind_order, fill_value=0)

                # 定義空氣品質順序（從良好到危害）
                quality_order = ['良好(0-50)', '普通(51-100)', '對敏感族群不健康(101-150)',
                                '不健康(151-200)', '非常不健康(201-300)', '危害(>300)']
                crosstab = crosstab.reindex(columns=quality_order, fill_value=0)

                # 轉換為整數
                crosstab = crosstab.astype(int)

                # 顯示交叉表
                st.dataframe(
                    crosstab,
                    use_container_width=True
                )

                st.caption("📊 數值為該組合的出現次數（天數），0 表示無數據")

                # Data summary
                with st.expander("📋 查看數據摘要"):
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.write("**風速統計 (m/s)**")
                        st.write(f"- 平均: {scatter_data['windspeed'].mean():.2f}")
                        st.write(f"- 中位數: {scatter_data['windspeed'].median():.2f}")
                        st.write(f"- 標準差: {scatter_data['windspeed'].std():.2f}")
                        st.write(f"- 範圍: {scatter_data['windspeed'].min():.2f} ~ {scatter_data['windspeed'].max():.2f}")

                    with summary_col2:
                        st.write("**AQI統計**")
                        st.write(f"- 平均: {scatter_data['aqi'].mean():.2f}")
                        st.write(f"- 中位數: {scatter_data['aqi'].median():.2f}")
                        st.write(f"- 標準差: {scatter_data['aqi'].std():.2f}")
                        st.write(f"- 範圍: {scatter_data['aqi'].min():.2f} ~ {scatter_data['aqi'].max():.2f}")

                    st.write(f"**樣本數**: {len(scatter_data)} 天")

            else:
                st.warning("⚠️ 數據不足，無法進行二維分析")
        else:
            st.error("❌ 缺少必要欄位 (windspeed 或 aqi)")

    with tab2:
        st.markdown("#### 按季節分組的關係分析")
        st.info("💡 觀察不同季節中 AQI 與風速的關係是否有差異")

        if 'windspeed' in df.columns and 'aqi' in df.columns and 'season' in df.columns:
            # Aggregate by date with season
            daily_data_season = df.groupby(df['date'].dt.date).agg({
                'windspeed': 'mean',
                'aqi': 'mean',
                'season': 'first'
            }).reset_index()
            daily_data_season.columns = ['date', 'windspeed', 'aqi', 'season']

            scatter_data_season = daily_data_season[['windspeed', 'aqi', 'season']].dropna()

            if len(scatter_data_season) > 0:
                # Define season order and color mapping for clear differentiation
                season_order = ['春季', '夏季', '秋季', '冬季']
                season_colors = {
                    '春季': '#32CD32',  # LimeGreen - bright, clearly green
                    '夏季': '#FF0000',  # Red - represents summer heat
                    '秋季': '#FFA500',  # Orange - represents autumn harvest
                    '冬季': '#0070C0'   # Blue - represents winter cold
                }

                # Ensure season column is categorical with specified order
                scatter_data_season['season'] = pd.Categorical(
                    scatter_data_season['season'],
                    categories=season_order,
                    ordered=True
                )

                # Create scatter plot with custom season colors
                fig = px.scatter(
                    scatter_data_season,
                    x='windspeed',
                    y='aqi',
                    color='season',
                    title='AQI 與風速的關係（按季節著色）',
                    labels={'windspeed': 'WINDSPEED', 'aqi': 'AQI', 'season': '季節'},
                    trendline="ols",
                    opacity=0.6,
                    color_discrete_map=season_colors,
                    category_orders={'season': season_order}
                )

                # Update layout for better visualization
                fig.update_layout(
                    template='plotly_white',
                    height=500,
                    hovermode='closest',
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=14,
                        font_family="Arial, Microsoft YaHei, sans-serif",
                        font_color="black"
                    )
                )

                # Update marker style
                fig.update_traces(
                    marker=dict(
                        size=8,
                        line=dict(width=1, color='white')
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Calculate correlation by season
                st.markdown("#### 各季節相關性分析")

                season_stats = []
                for season in scatter_data_season['season'].unique():
                    season_df = scatter_data_season[scatter_data_season['season'] == season]
                    if len(season_df) > 2:
                        from scipy.stats import pearsonr
                        corr_s, pval_s = pearsonr(season_df['windspeed'], season_df['aqi'])
                        season_stats.append({
                            '季節': season,
                            '樣本數': len(season_df),
                            '相關係數': f"{corr_s:.3f}",
                            'p-value': f"{pval_s:.4f}",
                            '平均風速': f"{season_df['windspeed'].mean():.2f}",
                            '平均AQI': f"{season_df['aqi'].mean():.2f}"
                        })

                if season_stats:
                    season_stats_df = pd.DataFrame(season_stats)
                    st.dataframe(season_stats_df, use_container_width=True)
            else:
                st.warning("⚠️ 數據不足，無法進行季節分析")
        else:
            st.error("❌ 缺少必要欄位 (windspeed, aqi 或 season)")

    with tab3:
        st.markdown("#### 其他污染物對比分析")

        pollutant_options = []
        for col in ['pm2.5', 'pm10', 'o3', 'co', 'so2', 'no2']:
            if col in df.columns:
                pollutant_options.append(col)

        if len(pollutant_options) >= 2:
            col1, col2 = st.columns(2)

            with col1:
                x_var = st.selectbox("選擇 X 軸變量", pollutant_options, key='x_var_scatter')

            with col2:
                y_var = st.selectbox(
                    "選擇 Y 軸變量",
                    [p for p in pollutant_options if p != x_var],
                    key='y_var_scatter'
                )

            # Aggregate by date
            daily_custom = df.groupby(df['date'].dt.date).agg({
                x_var: 'mean',
                y_var: 'mean'
            }).reset_index()
            daily_custom.columns = ['date', x_var, y_var]

            scatter_custom = daily_custom[[x_var, y_var]].dropna()

            if len(scatter_custom) > 0:
                # Calculate correlation
                from scipy.stats import pearsonr
                corr_c, pvalue_c = pearsonr(scatter_custom[x_var], scatter_custom[y_var])

                # Calculate slope
                import numpy as np
                z_c = np.polyfit(scatter_custom[x_var], scatter_custom[y_var], 1)
                slope_c = z_c[0]

                st.metric("回歸斜率", f"{slope_c:.2f}")

                # Create scatter plot
                fig = create_scatter_plot(
                    df=scatter_custom,
                    x_col=x_var,
                    y_col=y_var,
                    title=f'{y_var.upper()} vs {x_var.upper()}',
                    show_trendline=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 數據不足")
        else:
            st.warning("⚠️ 可用污染物欄位不足（需至少2個）")

    with tab4:
        st.markdown("#### 數據表格檢視")
        st.info("💡 顯示散點圖中每日平均數據的對應表格")

        if 'windspeed' in df.columns and 'aqi' in df.columns:
            # 使用與 Tab 1 相同的聚合邏輯
            daily_table_data = df.groupby(df['date'].dt.date).agg({
                'windspeed': 'mean',
                'aqi': 'mean'
            }).reset_index()

            # 格式化欄位名稱
            daily_table_data.columns = ['日期', '風速 (m/s)', 'AQI']

            # 移除缺失值
            daily_table_data = daily_table_data.dropna()

            # 數值格式化（保留2位小數）
            daily_table_data['風速 (m/s)'] = daily_table_data['風速 (m/s)'].round(2)
            daily_table_data['AQI'] = daily_table_data['AQI'].round(2)

            if len(daily_table_data) > 0:
                # 顯示表格
                st.dataframe(
                    daily_table_data,
                    use_container_width=True,
                    hide_index=True
                )

                # 數據摘要
                st.caption(f"📊 總資料筆數：{len(daily_table_data)} 天")
            else:
                st.warning("⚠️ 數據不足，無法顯示表格")
        else:
            st.error("❌ 缺少必要欄位 (windspeed 或 aqi)")

    # ===== Time Series Analysis =====
    st.markdown("---")
    st.subheader("3️⃣ 時間序列分析")

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
    st.subheader("4️⃣ 交叉表分析 (Crosstab)")

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

            # Define county order from North to South (geographic order) - English names
            # Based on app_utils.py region mapping
            county_order = [
                # 北部 (North)
                'Keelung City', 'Taipei City', 'New Taipei City', 'Taoyuan City',
                'Hsinchu City', 'Hsinchu County',
                # 中部 (Central)
                'Miaoli County', 'Taichung City', 'Changhua County',
                'Nantou County', 'Yunlin County',
                # 南部 (South)
                'Chiayi County', 'Chiayi City', 'Tainan City',
                'Kaohsiung City', 'Pingtung County',
                # 東部 (East)
                'Yilan County', 'Hualien County', 'Taitung County',
                # 離島 (Outlying Islands)
                'Penghu County', 'Kinmen County', 'Lienchiang County'
            ]

            # Reindex pivot_data to follow geographic order
            # Only include counties that actually exist in the data
            existing_counties = [c for c in county_order if c in pivot_data.index]
            pivot_data = pivot_data.reindex(existing_counties)

            # Create heatmap using plotly directly (to control county order)
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='RdYlGn_r',
                text=pivot_data.values.round(2),
                texttemplate='%{z:.2f}',
                textfont={"size": 10},
                colorbar=dict(title='AQI')
            ))

            fig.update_layout(
                title='各縣市各月份平均AQI熱力圖',
                xaxis_title='月份 (month)',
                yaxis_title='縣市 (county)',
                template='plotly_white',
                height=500,
                # Force yaxis to use custom geographic order (North to South)
                yaxis=dict(
                    categoryorder='array',
                    categoryarray=existing_counties
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14,
                    font_family="Arial, Microsoft YaHei, sans-serif",
                    font_color="black"
                )
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
    st.subheader("5️⃣ 分布分析")

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
    st.subheader("6️⃣ 空氣質量結構分析")

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

            # Use a line chart for clearer year-over-year trend.
            # If only one year is present (common when filtering a short date range),
            # fall back to showing that year's monthly trend instead of a single bar.
            n_years = year_structure['year'].nunique()
            if n_years >= 2:
                fig = px.line(
                    year_structure,
                    x='year',
                    y='平均AQI',
                    markers=True,
                    title='各年度平均AQI趨勢'
                )
                fig.update_traces(line=dict(color='#1f77b4', width=3))
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    hovermode='x unified',
                    xaxis=dict(dtick=1, title='年度'),
                    yaxis=dict(title='平均AQI')
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info('目前篩選僅含單一年度，改顯示該年度的月趨勢')
                monthly = df.groupby('month')['aqi'].mean().reset_index()
                fig = px.line(
                    monthly,
                    x='month',
                    y='aqi',
                    markers=True,
                    title='當年度各月份平均AQI'
                )
                fig.update_traces(line=dict(color='#1f77b4', width=3))
                fig.update_layout(
                    template='plotly_white',
                    height=400,
                    hovermode='x unified',
                    xaxis=dict(
                        title='月份',
                        tickmode='array',
                        tickvals=list(range(1, 13))
                    ),
                    yaxis=dict(title='平均AQI')
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
    """)
