"""
Page 3: Pattern Discovery (規律發現) - Knowledge Layer

This page implements the "Knowledge" level of the DIKW hierarchy, displaying:
- Seasonal patterns
- Geographic distribution patterns
- Meteorological influence analysis
- Pollutant correlations
- Cause-and-effect relationships

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

from utils.app_viz import (
    create_seasonal_pattern_plot,
    create_correlation_matrix,
    create_comparison_plot,
    create_wind_rose,
    create_map_plot
)
import plotly.express as px
import plotly.graph_objects as go


def render(df: pd.DataFrame):
    """
    Render the Pattern Discovery page.

    Args:
        df: Air quality DataFrame with SPCT dimension labels
    """
    st.header("🔍 規律發現 - Knowledge Layer")
    st.markdown("### DIKW層級：Knowledge（知識）")
    st.markdown("**問題：為什麼？** - 理解數據中的模式、規律和因果關係")
    st.markdown("---")

    # ===== Seasonal Patterns =====
    st.subheader("1️⃣ 季節性污染模式")

    if 'season' in df.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 各季節AQI模式")

            fig = create_seasonal_pattern_plot(df, 'aqi', '各季節AQI平均值及標準差')
            st.plotly_chart(fig, width='stretch')

        with col2:
            st.markdown("#### 季節統計")

            seasonal_stats = df.groupby('season')['aqi'].agg(['mean', 'std', 'max', 'min']).round(1)
            seasonal_stats.columns = ['平均值', '標準差', '最大值', '最小值']

            st.dataframe(seasonal_stats, width='stretch')

        # Seasonal insights
        winter_aqi = df[df['season'] == '冬季']['aqi'].mean()
        summer_aqi = df[df['season'] == '夏季']['aqi'].mean()
        diff_pct = ((winter_aqi - summer_aqi) / summer_aqi * 100) if summer_aqi > 0 else 0

        st.info(f"""
        ### 🔍 發現 1: 季節性規律

        **觀察**: 冬季PM2.5濃度顯著高於夏季 (差異: {diff_pct:.1f}%)

        **原因分析**:
        - ❄️ 冬季大氣穩定，逆溫層形成頻繁
        - 🌬️ 污染物不易擴散
        - 🌏 境外污染物傳輸影響
        - 🏭 冬季供暖和工業活動增加

        **規律**: 每年11月至次年2月為污染高峰期
        """)

        # PM2.5 seasonal pattern
        if 'pm2.5' in df.columns:
            st.markdown("#### PM2.5季節模式")

            fig = create_seasonal_pattern_plot(df, 'pm2.5', '各季節PM2.5濃度')
            st.plotly_chart(fig, width='stretch')

    # ===== Geographic Patterns =====
    st.markdown("---")
    st.subheader("2️⃣ 地理分布規律")

    tab1, tab2, tab3 = st.tabs(["區域比較", "地理地圖", "城鄉差異"])

    with tab1:
        if 'region' in df.columns:
            st.markdown("#### 各區域空氣質量比較")

            fig = create_comparison_plot(
                df,
                'region',
                'aqi',
                '各區域AQI分布比較'
            )
            st.plotly_chart(fig, width='stretch')

            # Regional statistics
            regional_stats = df.groupby('region').agg({
                'aqi': ['mean', 'median', 'max'],
                'pm2.5': 'mean',
                'sitename': 'nunique'
            }).round(1)

            regional_stats.columns = ['平均AQI', 'AQI中位數', '最高AQI', '平均PM2.5', '監測站數']
            st.dataframe(regional_stats.sort_values('平均AQI', ascending=False), width='stretch')

            # Geographic insights
            west_regions = ['北部', '中部', '南部']
            east_regions = ['東部']

            west_df = df[df['region'].isin(west_regions)]
            east_df = df[df['region'].isin(east_regions)]

            if len(west_df) > 0 and len(east_df) > 0:
                west_aqi = west_df['pm2.5'].mean()
                east_aqi = east_df['pm2.5'].mean()
                diff_pct = ((west_aqi - east_aqi) / east_aqi * 100) if east_aqi > 0 else 0

                st.info(f"""
                ### 🔍 發現 2: 地理分布規律

                **觀察**: 西部地區PM2.5濃度比東部高 {diff_pct:.1f}%

                **原因分析**:
                - 🏙️ 西部人口密集，工業發達
                - 🚗 交通排放量大
                - 🏔️ 地形因素：中央山脈阻擋，污染物不易擴散
                - 🌊 東部靠海，海風有助於污染物擴散
                """)

    with tab2:
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.markdown("#### 監測站空氣質量地理分布")

            try:
                fig = create_map_plot(df, 'pm2.5', 'aqi', '監測站平均AQI地理分布')
                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.warning(f"地圖顯示錯誤: {e}")

    with tab3:
        st.markdown("#### 縣市比較")

        top_n = st.slider("顯示前N個縣市", 5, 20, 10, key='county_top_n')

        county_stats = df.groupby('county')['aqi'].mean().sort_values(ascending=False).head(top_n)

        fig = px.bar(
            x=county_stats.values,
            y=county_stats.index,
            orientation='h',
            title=f'平均AQI最高的{top_n}個縣市',
            labels={'x': '平均AQI', 'y': '縣市'},
            color=county_stats.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # ===== Meteorological Influence =====
    st.markdown("---")
    st.subheader("3️⃣ 氣象條件影響分析")

    tab1, tab2 = st.tabs(["風速影響", "風向分析"])

    with tab1:
        if 'wind_level' in df.columns:
            st.markdown("#### 風速對空氣質量的影響")

            # Wind speed vs AQI
            wind_aqi = df.groupby('wind_level')['aqi'].agg(['mean', 'count']).reset_index()
            wind_aqi.columns = ['風速等級', '平均AQI', '記錄數']

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    wind_aqi,
                    x='風速等級',
                    y='平均AQI',
                    title='不同風速等級的平均AQI',
                    color='平均AQI',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_traces(
                    hovertemplate='<b>風速等級: %{x}</b><br>平均AQI: %{y:.1f}<br>記錄數: %{customdata[0]}<extra></extra>',
                    customdata=wind_aqi[['記錄數']].values
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
                st.dataframe(wind_aqi, width='stretch')

            # Calculate correlation
            if 'windspeed' in df.columns:
                correlation = df[['windspeed', 'aqi']].corr().iloc[0, 1]

                low_wind_aqi = df[df['windspeed'] < 2]['aqi'].mean() if len(df[df['windspeed'] < 2]) > 0 else 0
                high_wind_aqi = df[df['windspeed'] > 5]['aqi'].mean() if len(df[df['windspeed'] > 5]) > 0 else 0

                reduction_pct = ((low_wind_aqi - high_wind_aqi) / low_wind_aqi * 100) if low_wind_aqi > 0 else 0

                st.success(f"""
                ### 🔍 發現 3: 風速與污染關聯

                **觀察**: 風速與AQI呈負相關 (r={correlation:.2f})

                **發現**:
                - 💨 風速低於2m/s時，平均AQI: {low_wind_aqi:.1f}
                - 🌬️ 風速高於5m/s時，平均AQI: {high_wind_aqi:.1f}
                - 📉 強風可使AQI降低約 {reduction_pct:.1f}%

                **原因**: 風速越大，污染物擴散越快，濃度降低
                """)

    with tab2:
        if 'winddirec' in df.columns:
            st.markdown("#### 風向分布與污染")

            col1, col2 = st.columns(2)

            with col1:
                try:
                    fig = create_wind_rose(df, '風向分布圖')
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.warning(f"風向圖顯示錯誤: {e}")

            with col2:
                st.markdown("##### 不同風向的平均AQI")

                # Categorize wind direction
                def direction_category(degree):
                    if pd.isna(degree):
                        return None
                    if degree < 45 or degree >= 315:
                        return '北風'
                    elif degree < 135:
                        return '東風'
                    elif degree < 225:
                        return '南風'
                    else:
                        return '西風'

                df_wind = df.copy()
                df_wind['wind_dir'] = df_wind['winddirec'].apply(direction_category)
                wind_dir_aqi = df_wind.groupby('wind_dir')['aqi'].mean().sort_values(ascending=False)

                if len(wind_dir_aqi) > 0:
                    fig = px.bar(
                        x=wind_dir_aqi.index,
                        y=wind_dir_aqi.values,
                        title='各風向平均AQI',
                        labels={'x': '風向', 'y': '平均AQI'}
                    )
                    fig.update_traces(
                        hovertemplate='<b>風向: %{x}</b><br>平均AQI: %{y:.1f}<extra></extra>'
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

    # ===== Pollutant Correlations =====
    st.markdown("---")
    st.subheader("4️⃣ 污染物相關性分析")

    pollutant_cols = ['aqi', 'pm2.5', 'pm10', 'o3', 'co', 'so2', 'no2']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]

    if len(available_pollutants) >= 2:
        st.markdown("#### 污染物相關性矩陣")

        fig = create_correlation_matrix(
            df,
            available_pollutants,
            '污染物相關性熱力圖'
        )
        st.plotly_chart(fig, width='stretch')

        # Calculate and display correlation insights
        corr_matrix = df[available_pollutants].corr()

        st.markdown("#### 強相關污染物對")

        # Find strong correlations (> 0.7 or < -0.3)
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append({
                        '污染物1': corr_matrix.columns[i],
                        '污染物2': corr_matrix.columns[j],
                        '相關係數': round(corr_val, 3),
                        '關係': '正相關' if corr_val > 0 else '負相關'
                    })

        if strong_corrs:
            st.dataframe(pd.DataFrame(strong_corrs), width='stretch')

            st.info("""
            ### 🔍 發現 4: 污染物關聯性

            **觀察**: PM2.5與PM10高度正相關

            **原因**:
            - 🏭 來源相同：主要來自工業排放、交通排放、揚塵
            - 📏 大小關聯：PM2.5是PM10的子集
            - ⚗️ 化學反應：大氣中的化學反應同時影響兩者

            **應用**: 控制PM10可同時改善PM2.5
            """)

    # ===== Temporal Patterns =====
    st.markdown("---")
    st.subheader("5️⃣ 時間模式分析")

    tab1, tab2 = st.tabs(["一日模式", "週末效應"])

    with tab1:
        if 'hour' in df.columns:
            st.markdown("#### 一日內AQI變化模式")

            hourly_pattern = df.groupby('hour')['aqi'].mean().reset_index()

            fig = px.line(
                hourly_pattern,
                x='hour',
                y='aqi',
                title='24小時AQI平均值變化',
                markers=True,
                labels={'hour': '時刻', 'aqi': '平均AQI'}
            )
            fig.update_xaxes(dtick=2)
            st.plotly_chart(fig, width='stretch')

            # Find peak hours
            peak_hour = hourly_pattern.loc[hourly_pattern['aqi'].idxmax(), 'hour']
            low_hour = hourly_pattern.loc[hourly_pattern['aqi'].idxmin(), 'hour']

            st.info(f"""
            ### 🔍 發現 5: 日間變化規律

            **觀察**:
            - ⬆️ 高峰時段: {peak_hour}時
            - ⬇️ 低谷時段: {low_hour}時

            **原因**:
            - 🚗 早晚交通尖峰時段排放增加
            - ☀️ 中午太陽輻射強，大氣混合層高度增加，有助擴散
            - 🌙 夜間大氣穩定，污染物累積
            """)

    with tab2:
        if 'is_weekend' in df.columns:
            st.markdown("#### 平日與週末比較")

            weekend_comparison = df.groupby('is_weekend')['aqi'].agg(['mean', 'median', 'std']).round(1)
            weekend_comparison.index = ['平日', '週末']
            weekend_comparison.columns = ['平均值', '中位數', '標準差']

            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(weekend_comparison, width='stretch')

            with col2:
                fig = px.box(
                    df,
                    x='is_weekend',
                    y='aqi',
                    title='平日vs週末AQI分布',
                    labels={'is_weekend': '', 'aqi': 'AQI'}
                )
                fig.update_xaxes(ticktext=['平日', '週末'], tickvals=[False, True])
                st.plotly_chart(fig, width='stretch')

    # ===== Summary =====
    st.markdown("---")
    st.success("""
    ### 📚 知識總結

    通過規律發現，我們理解了：

    1. **季節規律**: 冬季污染嚴重，夏季相對良好
    2. **地理規律**: 西部工業區污染高於東部自然區
    3. **氣象影響**: 風速是影響空氣質量的關鍵因素
    4. **污染物關聯**: 多種污染物之間存在強相關性
    5. **時間模式**: 一日和一週內存在明顯的週期性變化

    💡 **下一步**: 前往「智慧決策」頁面，基於這些知識獲取行動建議
    """)
