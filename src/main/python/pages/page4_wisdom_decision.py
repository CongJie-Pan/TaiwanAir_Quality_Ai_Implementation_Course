"""
Page 4: Wisdom Decision (智慧決策) - Wisdom Layer

This page implements the "Wisdom" level of the DIKW hierarchy, providing:
- Current air quality status and alerts
- Health recommendations by user group
- Simple forecast (moving average)
- Policy suggestions for government
- Actionable decision support

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

from utils.app_utils import get_aqi_color, get_aqi_recommendation
import plotly.express as px
import plotly.graph_objects as go


def render(df: pd.DataFrame):
    """
    Render the Wisdom Decision page.

    Args:
        df: Air quality DataFrame with SPCT dimension labels
    """
    st.header("🎯 智慧決策 - Wisdom Layer")
    st.markdown("### DIKW層級：Wisdom（智慧）")
    st.markdown("**問題：怎麼做？** - 基於知識的預測、建議和決策支持")
    st.markdown("---")

    # ===== Current Air Quality Status =====
    st.subheader("1️⃣ 當前空氣質量狀態")

    # Get latest data
    latest_data = df.sort_values('date', ascending=False).iloc[0]
    current_aqi = latest_data['aqi']
    current_date = latest_data['date']
    current_county = latest_data['county']
    current_station = latest_data['sitename']

    # Determine AQI level and color
    aqi_color = get_aqi_color(current_aqi)

    if current_aqi <= 50:
        aqi_level = "良好"
        icon = "🟢"
    elif current_aqi <= 100:
        aqi_level = "普通"
        icon = "🟡"
    elif current_aqi <= 150:
        aqi_level = "對敏感族群不健康"
        icon = "🟠"
    elif current_aqi <= 200:
        aqi_level = "不健康"
        icon = "🔴"
    elif current_aqi <= 300:
        aqi_level = "非常不健康"
        icon = "🟣"
    else:
        aqi_level = "危害"
        icon = "🟤"

    # Display current status in a prominent box
    st.markdown(f"""
    <div style='background-color: {aqi_color}; padding: 30px; border-radius: 10px; text-align: center;'>
        <h1 style='color: white; margin: 0;'>{icon} 當前空氣品質：{aqi_level}</h1>
        <h2 style='color: white; margin: 10px 0;'>AQI: {current_aqi:.0f}</h2>
        <p style='color: white; font-size: 18px; margin: 5px 0;'>
            📍 {current_county} - {current_station}<br>
            📅 {current_date.strftime('%Y-%m-%d %H:%M')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Show pollutant breakdown
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pm25_val = latest_data.get('pm2.5', 0)
        st.metric("PM2.5", f"{pm25_val:.1f} μg/m³")

    with col2:
        pm10_val = latest_data.get('pm10', 0)
        st.metric("PM10", f"{pm10_val:.1f} μg/m³")

    with col3:
        o3_val = latest_data.get('o3', 0)
        st.metric("O3", f"{o3_val:.1f} ppb")

    with col4:
        main_pollutant = latest_data.get('pollutant', 'N/A')
        st.metric("主要污染物", main_pollutant)

    # ===== Health Recommendations =====
    st.markdown("---")
    st.subheader("2️⃣ 個人化健康建議")

    # User group selection
    col1, col2 = st.columns([1, 2])

    with col1:
        user_group = st.selectbox(
            "選擇您的身份",
            ["一般民眾", "敏感族群", "戶外工作者", "運動愛好者"],
            help="根據不同身份提供針對性建議"
        )

    with col2:
        # Get recommendation
        recommendation = get_aqi_recommendation(current_aqi, user_group)

        st.info(f"""
        ### 💡 針對 {user_group} 的建議

        {recommendation}
        """)

    # Detailed recommendations by scenario
    st.markdown("#### 詳細行動建議")

    tab1, tab2, tab3, tab4 = st.tabs(["日常活動", "運動建議", "防護措施", "交通出行"])

    with tab1:
        st.markdown("##### 🏠 日常活動建議")

        if current_aqi <= 100:
            st.success("""
            ✅ **可正常進行所有日常活動**
            - 開窗通風保持室內空氣流通
            - 適合戶外休閒活動
            - 無需特別防護措施
            """)
        elif current_aqi <= 150:
            st.warning("""
            ⚠️ **適度調整活動**
            - 敏感族群考慮減少戶外活動時間
            - 減少開窗時間，使用空氣清淨機
            - 兒童和老人避免長時間戶外活動
            """)
        else:
            st.error("""
            🚫 **建議留在室內**
            - 避免不必要的外出
            - 關閉門窗，使用空氣清淨機
            - 必要外出時配戴N95口罩
            - 減少室內劇烈活動
            """)

    with tab2:
        st.markdown("##### 🏃 運動建議")

        if current_aqi <= 50:
            st.success("""
            ✅ **適合所有運動**
            - 可進行高強度戶外運動
            - 馬拉松、騎行、球類運動均可
            - 最佳運動時間：早晨或傍晚
            """)
        elif current_aqi <= 100:
            st.success("""
            ✅ **可正常運動，略作調整**
            - 可進行中等強度運動
            - 減少超長時間戶外運動
            - 敏感族群選擇室內運動
            """)
        elif current_aqi <= 150:
            st.warning("""
            ⚠️ **減少戶外運動**
            - 改為室內運動（健身房、瑜伽等）
            - 若需戶外運動，縮短時間並降低強度
            - 敏感族群避免所有戶外運動
            """)
        else:
            st.error("""
            🚫 **停止戶外運動**
            - 取消所有戶外運動計劃
            - 改為室內輕度運動或休息
            - 運動後多補充水分
            """)

    with tab3:
        st.markdown("##### 😷 防護措施")

        if current_aqi <= 100:
            st.info("✅ 無需特別防護措施")
        else:
            st.warning("""
            ### 建議防護措施：

            **1. 口罩選擇**
            - 😷 AQI 101-150: 一般醫用口罩或KN95
            - 😷 AQI 151-200: N95或KN95口罩
            - 😷 AQI > 200: N95口罩，縮短戶外時間

            **2. 室內防護**
            - 🏠 關閉門窗
            - 🌀 使用空氣清淨機（HEPA濾網）
            - 🌿 室內放置空氣淨化植物

            **3. 飲食調理**
            - 🥗 多吃新鮮蔬果
            - 💧 多喝水幫助代謝
            - 🫖 飲用清熱潤肺茶飲
            """)

    with tab4:
        st.markdown("##### 🚗 交通出行建議")

        if current_aqi <= 100:
            st.success("""
            ✅ **可正常出行**
            - 推薦步行或騎自行車
            - 享受戶外空氣
            """)
        else:
            st.warning("""
            ⚠️ **調整出行方式**

            **建議**:
            - 🚇 優先選擇地鐵、公車等大眾運輸
            - 🚗 開車時使用車內空氣循環模式
            - 🚴 避免騎自行車或機車（會吸入更多污染物）
            - ⏰ 避開交通尖峰時段出行

            **開車族**:
            - 關閉車窗，使用車內空調循環
            - 更換車內空調濾網
            - 避免在車內久留
            """)

    # ===== Simple Forecast =====
    st.markdown("---")
    st.subheader("3️⃣ 未來趨勢預測（簡易模型）")

    # Calculate simple moving average forecast
    forecast_days = 3

    # Aggregate daily data
    daily_aqi = df.groupby(df['date'].dt.date)['aqi'].mean().reset_index()
    daily_aqi.columns = ['date', 'aqi']
    daily_aqi['date'] = pd.to_datetime(daily_aqi['date'])
    daily_aqi = daily_aqi.sort_values('date')

    # Calculate 7-day moving average
    window = min(7, len(daily_aqi))
    if window > 0:
        ma_value = daily_aqi['aqi'].tail(window).mean()

        # Generate forecast dates
        last_date = daily_aqi['date'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': [ma_value] * forecast_days,
            'lower': [ma_value * 0.85] * forecast_days,
            'upper': [ma_value * 1.15] * forecast_days
        })

        # Plot historical and forecast
        fig = go.Figure()

        # Historical data
        historical_window = daily_aqi.tail(30)
        fig.add_trace(go.Scatter(
            x=historical_window['date'],
            y=historical_window['aqi'],
            mode='lines',
            name='歷史數據',
            line=dict(color='blue', width=2)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            mode='lines+markers',
            name='預測值',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name='信賴區間',
            showlegend=True
        ))

        # Add threshold lines
        fig.add_hline(y=100, line_dash="dash", line_color="orange",
                     annotation_text="普通上限(100)")

        fig.update_layout(
            title='AQI預測（基於7日移動平均）',
            xaxis_title='日期',
            yaxis_title='AQI',
            hovermode='x unified',
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, width='stretch')

        # Display forecast values
        st.markdown("##### 預測結果")

        col1, col2, col3 = st.columns(3)

        for i, (date, aqi_pred, lower, upper) in enumerate(zip(
            forecast_df['date'], forecast_df['forecast'],
            forecast_df['lower'], forecast_df['upper']
        )):
            with [col1, col2, col3][i]:
                pred_color = get_aqi_color(aqi_pred)
                st.markdown(f"""
                <div style='background-color: {pred_color}; padding: 15px; border-radius: 5px; text-align: center;'>
                    <p style='color: white; margin: 0; font-size: 14px;'>{date.strftime('%m/%d (%a)')}</p>
                    <h3 style='color: white; margin: 5px 0;'>{aqi_pred:.0f}</h3>
                    <p style='color: white; margin: 0; font-size: 12px;'>{lower:.0f} - {upper:.0f}</p>
                </div>
                """, unsafe_allow_html=True)

        st.info("""
        📊 **預測方法說明**:
        - 使用7日移動平均法進行簡易預測
        - 信賴區間為預測值的±15%
        - 此為簡易模型，實際空氣質量受多種因素影響
        - 建議參考官方預報獲取更準確資訊
        """)

    # ===== Policy Recommendations =====
    st.markdown("---")
    st.subheader("4️⃣ 政策建議與應變措施")

    # Calculate average AQI for policy decisions
    recent_aqi = daily_aqi['aqi'].tail(7).mean() if len(daily_aqi) > 0 else current_aqi
    exceed_days = len(daily_aqi[daily_aqi['aqi'] > 100].tail(10))

    tab1, tab2, tab3 = st.tabs(["即時應變", "中長期政策", "監測建議"])

    with tab1:
        st.markdown("##### 🚨 即時應變措施（針對環保單位）")

        if recent_aqi <= 100:
            st.success("""
            ### ✅ 空氣品質良好，維持現狀

            **持續監測**:
            - 保持日常監測頻率
            - 關注氣象變化
            - 準備應變機制
            """)
        elif recent_aqi <= 150:
            st.warning("""
            ### ⚠️ 啟動第一級應變措施

            **建議措施**:
            1. 📢 發布空氣品質提醒
            2. 🏫 通知學校減少戶外活動
            3. 🏭 要求重點排放源自主減排
            4. 🚧 加強道路清掃抑制揚塵
            5. 🔍 增加監測頻率

            **執行單位**: 環保局、教育局
            """)
        else:
            st.error(f"""
            ### 🚫 啟動嚴重污染應變

            **緊急措施** (近7日平均AQI: {recent_aqi:.0f}):

            1. 📢 **發布健康警報**
               - 透過媒體、APP、簡訊廣播
               - 提醒民眾減少外出

            2. 🏭 **工業排放管制**
               - 重點污染源降載減排20-30%
               - 禁止露天燃燒
               - 加強稽查力度

            3. 🚗 **交通管制**
               - 鼓勵大眾運輸，提供優惠
               - 考慮車輛分流管制
               - 大型車輛禁止進入市區

            4. 🏗️ **工程管制**
               - 停止土石方作業
               - 加強工地灑水抑塵
               - 暫停非必要工程

            5. 🏫 **學校措施**
               - 停止戶外課程
               - 關閉門窗、開啟空氣清淨機
               - 必要時考慮停課

            **執行單位**: 環保局、警察局、教育局、工務局
            """)

    with tab2:
        st.markdown("##### 🎯 中長期改善政策建議")

        st.info("""
        ### 永續發展策略

        **1. 能源轉型** 🔋
        - 推動再生能源使用
        - 淘汰老舊燃煤電廠
        - 提高能源效率標準

        **2. 交通革新** 🚇
        - 擴大大眾運輸覆蓋率
        - 推廣電動車輛
        - 建設自行車道網絡
        - 實施綠色物流

        **3. 工業升級** 🏭
        - 推動清潔生產技術
        - 提高排放標準
        - 實施總量管制
        - 鼓勵企業使用低污染能源

        **4. 都市規劃** 🌳
        - 增加綠地和公園
        - 建設綠色廊道
        - 推動綠建築
        - 優化都市通風

        **5. 區域合作** 🤝
        - 跨縣市聯防聯控
        - 建立區域預警系統
        - 共享治理經驗
        - 協同應對境外污染
        """)

    with tab3:
        st.markdown("##### 📊 監測網絡優化建議")

        # Analyze monitoring coverage
        if 'county' in df.columns:
            county_coverage = df.groupby('county')['sitename'].nunique().sort_values()

            st.write("**各縣市監測站數量:**")
            st.bar_chart(county_coverage)

        st.info("""
        ### 監測網絡改善建議

        **1. 擴大監測覆蓋**
        - 🎯 在監測站較少的區域增設站點
        - 🏘️ 在人口密集區加密監測
        - 🏭 在重點污染源周邊設置監測

        **2. 技術升級**
        - 📱 部署微型感測器
        - 🛰️ 整合衛星遙測數據
        - 🤖 引入AI預測模型
        - 📊 建立即時數據平台

        **3. 數據應用**
        - 🔓 開放數據供研究使用
        - 📲 開發民眾查詢APP
        - 📧 建立預警通知系統
        - 📈 定期發布分析報告
        """)

    # ===== Summary =====
    st.markdown("---")
    st.success(f"""
    ### 🎯 決策支持總結

    **當前狀態**: AQI {current_aqi:.0f} - {aqi_level}

    **立即行動**:
    - 👤 個人: {get_aqi_recommendation(current_aqi, user_group)}
    - 🏛️ 政府: {'維持現狀監測' if current_aqi <= 100 else '啟動應變措施'}

    **預測趨勢**: 未來3天AQI預測約 {ma_value:.0f}

    💡 **關鍵建議**: 根據DIKW分析，空氣質量受季節、氣象和人為活動多重影響。
    建議採取「預防為主、應急為輔」策略，從源頭減少污染排放。
    """)
