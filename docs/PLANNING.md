# 台灣空氣質量數據分析系統 - 專案規劃書

**Project:** Air Quality Analysis System with Streamlit
**Based on:** DIKW Principle & Multi-Dimensional Analysis (MDA)
**Course Reference:** AIp04 空間與網站設計
**Created:** 2025-10-13
**Version:** 1.0

---

## 目錄

1. [專案概述](#1-專案概述)
2. [數據維度分析（MDA）](#2-數據維度分析mda)
3. [DIKW原則架構](#3-dikw原則架構)
4. [Streamlit系統設計](#4-streamlit系統設計)
5. [技術實現細節](#5-技術實現細節)
6. [開發路線圖](#6-開發路線圖)

---

## 1. 專案概述

### 1.1 數據來源與規模

**數據基本信息：**
- **來源：** 台灣環保署空氣質量監測網
- **時間範圍：** 2016-11-25 至 2024-08-31（約8年）
- **數據規模：** 5,882,208 筆記錄
- **地理覆蓋：** 123個監測站，遍布台灣22個縣市
- **測量頻率：** 每小時記錄

**數據格式優化成果：**
```
原始格式：CSV (801 MB)
優化格式：Parquet (115 MB) - 壓縮率 85.7%
數據庫：DuckDB (175 MB) - 支援SQL查詢
性能提升：查詢速度提升 20-50倍
```

### 1.2 數據欄位結構（25個欄位）

**維度欄位（Dimensions）：**
- **時間維度（T）：** `date` (datetime)
- **空間維度（S）：** `sitename`, `county`, `longitude`, `latitude`, `siteid`
- **污染物維度（P）：** `pollutant` (主要污染物類型)
- **狀態維度（C）：** `status` (空氣質量狀態)

**量值欄位（Measures）：**

| 類別 | 欄位 | 單位 | 說明 |
|------|------|------|------|
| **懸浮微粒** | pm2.5 | μg/m³ | PM2.5細懸浮微粒 |
| | pm10 | μg/m³ | PM10懸浮微粒 |
| | pm2.5_avg | μg/m³ | PM2.5移動平均 |
| | pm10_avg | μg/m³ | PM10移動平均 |
| **氣態污染物** | o3 | ppb | 臭氧 |
| | o3_8hr | ppb | 臭氧8小時平均 |
| | co | ppm | 一氧化碳 |
| | co_8hr | ppm | CO 8小時平均 |
| | so2 | ppb | 二氧化硫 |
| | so2_avg | ppb | SO2平均值 |
| | no2 | ppb | 二氧化氮 |
| | nox | ppb | 氮氧化物 |
| | no | ppb | 一氧化氮 |
| **綜合指標** | aqi | - | 空氣質量指數 |
| **氣象參數** | windspeed | m/s | 風速 |
| | winddirec | degree | 風向 |

### 1.3 專案目標

**核心目標：**
1. ✅ 建立高效的數據載入與查詢系統（已完成）
2. 🎯 實現多維度數據分析（SPCT模型）
3. 🎯 展示DIKW原則的實際應用
4. 🎯 開發互動式Streamlit儀表板
5. 🎯 提供空氣質量洞察與預測

**預期產出：**
- 5頁專案簡報（PPT）
- 互動式Streamlit網站系統
- 數據分析報告
- 技術文檔

---

## 2. 數據維度分析（MDA）

### 2.1 數據空間的四維模型（SPCT）

參照 **AIp04空間與網站A.py** 的SPC模型，針對空氣質量數據調整為SPCT模型：

```
原始模型（零售業）:
S (銷售) - P (商品) - C (客戶) - T (時間)

空氣質量模型:
S (空間) - P (污染物) - C (條件) - T (時間)
```

#### 2.1.1 S維度：空間/地點層次結構

**層次結構（Hierarchy）：**
```
地理區域 → 縣市 → 鄉鎮區 → 監測站
```

**具體分類：**

| 區域 | 縣市 | 代表站點 |
|------|------|----------|
| **北部區域** | 台北市、新北市、基隆市、桃園市、新竹縣市 | 中山、板橋、桃園、新竹 |
| **中部區域** | 台中市、彰化縣、南投縣、苗栗縣、雲林縣 | 豐原、彰化、南投 |
| **南部區域** | 高雄市、台南市、屏東縣、嘉義縣市 | 前金、安南、屏東 |
| **東部區域** | 花蓮縣、台東縣 | 花蓮、台東 |
| **離島區域** | 澎湖縣、金門縣、連江縣 | 馬公、金門 |

**S維度的衍生屬性：**
```python
# 區域分類
df['region'] = df['county'].map({
    '台北市': '北部', '新北市': '北部', '基隆市': '北部',
    '台中市': '中部', '彰化縣': '中部', '南投縣': '中部',
    '高雄市': '南部', '台南市': '南部', '屏東縣': '南部',
    '花蓮縣': '東部', '台東縣': '東部',
    # ... 其他縣市
})

# 都市類型
df['urban_type'] = df['county'].map({
    '台北市': '都會區', '新北市': '都會區', '台中市': '都會區',
    # ... 其他分類
})
```

#### 2.1.2 P維度：污染物分類層次

**層次結構：**
```
污染類別 → 污染物種類 → 測量指標
```

**污染物分類樹：**

```
空氣污染物
├─ 懸浮微粒類 (Particulate Matter)
│  ├─ PM2.5 (細懸浮微粒)
│  │  ├─ pm2.5 (即時值)
│  │  └─ pm2.5_avg (移動平均)
│  └─ PM10 (懸浮微粒)
│     ├─ pm10 (即時值)
│     └─ pm10_avg (移動平均)
│
├─ 氣態污染物 (Gaseous Pollutants)
│  ├─ 臭氧 (O3)
│  │  ├─ o3 (即時值)
│  │  └─ o3_8hr (8小時平均)
│  ├─ 一氧化碳 (CO)
│  │  ├─ co (即時值)
│  │  └─ co_8hr (8小時平均)
│  ├─ 二氧化硫 (SO2)
│  │  ├─ so2 (即時值)
│  │  └─ so2_avg (平均值)
│  └─ 氮氧化物 (NOx)
│     ├─ no2 (二氧化氮)
│     ├─ no (一氧化氮)
│     └─ nox (總氮氧化物)
│
└─ 綜合指標
   └─ AQI (空氣質量指數)
      ├─ 良好 (0-50)
      ├─ 普通 (51-100)
      ├─ 對敏感族群不健康 (101-150)
      ├─ 不健康 (151-200)
      ├─ 非常不健康 (201-300)
      └─ 危害 (301+)
```

**P維度的衍生標籤：**
```python
# 污染物分類標籤
df['pollutant_category'] = df['pollutant'].map({
    'PM2.5': '懸浮微粒',
    'PM10': '懸浮微粒',
    'O3': '氣態污染物',
    'CO': '氣態污染物',
    'SO2': '氣態污染物',
    'NO2': '氣態污染物'
})

# AQI等級標籤
def aqi_level(aqi):
    if aqi <= 50: return '良好'
    elif aqi <= 100: return '普通'
    elif aqi <= 150: return '對敏感族群不健康'
    elif aqi <= 200: return '不健康'
    elif aqi <= 300: return '非常不健康'
    else: return '危害'

df['aqi_level'] = df['aqi'].apply(aqi_level)
```

#### 2.1.3 C維度：條件/情境分類

**層次結構：**
```
環境條件 → 氣象條件 → 空氣品質狀態
```

**C維度的分類體系：**

1. **季節條件：**
   - 春季 (3-5月)
   - 夏季 (6-8月)
   - 秋季 (9-11月)
   - 冬季 (12-2月)

2. **氣象條件：**
   ```
   風速條件
   ├─ 無風 (< 1 m/s)
   ├─ 微風 (1-3 m/s)
   ├─ 輕風 (3-5 m/s)
   └─ 強風 (> 5 m/s)

   風向條件
   ├─ 北風 (315-45°)
   ├─ 東風 (45-135°)
   ├─ 南風 (135-225°)
   └─ 西風 (225-315°)
   ```

3. **時段條件：**
   - 早晨 (06:00-09:00)
   - 上午 (09:00-12:00)
   - 下午 (12:00-18:00)
   - 傍晚 (18:00-21:00)
   - 夜間 (21:00-06:00)

**C維度的衍生標籤：**
```python
# 季節標籤
df['season'] = pd.cut(df['month'],
                      bins=[0,3,6,9,12],
                      labels=['冬季','春季','夏季','秋季'])

# 風速等級
df['wind_level'] = pd.cut(df['windspeed'],
                          bins=[0,1,3,5,float('inf')],
                          labels=['無風','微風','輕風','強風'])

# 工作日/週末
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 時段標籤
df['time_period'] = pd.cut(df['date'].dt.hour,
                           bins=[0,6,9,12,18,21,24],
                           labels=['夜間','早晨','上午','下午','傍晚','夜間'],
                           include_lowest=True)
```

#### 2.1.4 T維度：時間層次結構

**層次結構：**
```
年 (Year) → 季 (Quarter) → 月 (Month) → 週 (Week) → 日 (Day) → 時 (Hour)
```

**時間維度的詳細分解：**

| 層級 | 欄位名稱 | 範例值 | 說明 |
|------|----------|--------|------|
| **年** | `year` | 2024 | 2016-2024 |
| **季** | `quarter` | Q3 | 1-4季 |
| **年季** | `yq` | 24Q3 | 年+季組合 |
| **月** | `month` | 8 | 1-12月 |
| **年月** | `ym` | 2024-08 | 年+月組合 |
| **週** | `week` | 35 | 年度第幾週 (1-53) |
| **日** | `day` | 31 | 月份中的第幾天 |
| **星期** | `dayofweek` | 6 | 週一=0, 週日=6 |
| **時** | `hour` | 23 | 0-23時 |

**T維度的衍生標籤生成：**
```python
# 參照 AIp04/A3 數據標籤的產生方式
# (1) 日期相關標籤
df['year'] = pd.to_datetime(df['date']).dt.year
df['month'] = pd.to_datetime(df['date']).dt.month
df['day'] = pd.to_datetime(df['date']).dt.day
df['hour'] = pd.to_datetime(df['date']).dt.hour
df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek

# (2) 連續數值的離散化 - 季度標籤
df['quarter'] = pd.cut(df['month'],
                       bins=[0,3,6,9,12],
                       labels=['Q1','Q2','Q3','Q4'])

# (3) 文字處理的標籤 - 年季組合
df['yq'] = df['year'].astype(str).str[-2:] + 'Q' + df['quarter'].astype(str)

# (4) 年月標籤
df['ym'] = df['date'].dt.to_period('M')

# (5) 工作日標籤
df['is_workday'] = ~df['date'].dt.dayofweek.isin([5,6])
```

### 2.2 量值（Measures）與統計量

參照 **AIp04/A4 量值** 的定義：

#### 2.2.1 空氣質量專用量值

**定義類似零售業的"成交結構"：**

```python
# 參照 AIp04 的成交結構函數
def 空氣質量結構(df, group_by='county'):
    """
    計算空氣質量的結構性指標

    類比零售業成交結構:
    營業額 → 總AQI積分
    來客數 → 監測站數
    交易數 → 測量次數
    客單價 → 平均AQI
    """
    result = df.groupby(group_by).agg({
        'sitename': 'nunique',      # 監測站數
        'date': 'count',            # 測量次數
        'aqi': ['mean', 'median', 'max', 'min', 'sum'],
        'pm2.5': 'mean',
        'pm10': 'mean',
        'o3': 'mean'
    }).reset_index()

    result.columns = [
        group_by,
        '監測站數', '測量次數',
        '平均AQI', 'AQI中位數', '最高AQI', '最低AQI', 'AQI總積分',
        '平均PM2.5', '平均PM10', '平均O3'
    ]

    # 計算衍生指標
    result['污染天數佔比'] = (result['最高AQI'] > 100).astype(float)
    result['站均測量次數'] = result['測量次數'] / result['監測站數']

    return result
```

**空氣質量專用KPI：**

| KPI名稱 | 計算公式 | 說明 | 目標值 |
|---------|----------|------|--------|
| **達標率** | (AQI≤100的天數/總天數)×100% | 空氣品質良好的比例 | >80% |
| **超標天數** | count(AQI>100) | 空氣品質不佳的天數 | <20天/年 |
| **污染物超標次數** | count(污染物>標準值) | 各污染物超標情況 | 0次 |
| **連續污染天數** | max(連續AQI>100) | 最長污染持續時間 | <5天 |
| **污染改善率** | (本期-上期)/上期×100% | 同比污染改善情況 | >0% |
| **站點達標率** | 達標站點數/總站點數 | 區域整體空氣品質 | >90% |

### 2.3 多維度分析（MDA）的標準二維表格

參照 **AIp04/B2 多維度分析中的二維表格**：

#### 交叉表分析 (Crosstab)

```python
# (1) 頻次分布表 - 各年度各縣市的測量次數
TYC = pd.crosstab(df['year'], df['county'], margins=True)

# (2) 數值累計表 - 各年度各縣市的平均AQI
TYCM = pd.crosstab(
    index=df['year'],
    columns=df['county'],
    values=df['aqi'],
    aggfunc='mean',
    margins=True
)

# (3) 比例分布表 - 各污染等級在各縣市的分布比例
TYCMP = round(100 * pd.crosstab(
    index=df['aqi_level'],
    columns=df['county'],
    normalize='columns'
), 1)
```

#### 數據樞紐轉換 (Pivot)

**SPCT軸向投射：**

```python
# 參照 AIp04/B3 數據框轉換/樞紐(pivot)轉換

# (1) 投射到 S軸 (空間) + 時間(year)
Sv = df.groupby(['year', 'county']).agg({
    'aqi': 'mean',
    'pm2.5': 'mean',
    'sitename': 'nunique'
}).reset_index()
Sv.columns = ['year', 'county', 'avg_aqi', 'avg_pm25', 'station_count']

# (2) 投射到 P軸 (污染物) + 時間(year)
Pv = df.groupby(['year', 'pollutant']).agg({
    'date': 'count',
    'aqi': 'mean'
}).reset_index()
Pv.columns = ['year', 'pollutant', 'occurrence_count', 'avg_aqi']

# (3) 投射到 C軸 (條件) + 時間(month)
Cv = df.groupby(['month', 'season', 'wind_level']).agg({
    'aqi': ['mean', 'max'],
    'pm2.5': 'mean'
}).reset_index()

# (4) 投射到 T軸 (時間) - 年季分析
Tv = df.groupby(['yq']).agg({
    'aqi': 'mean',
    'pm2.5': 'mean',
    'sitename': 'nunique'
}).reset_index()
```

---

## 3. DIKW原則架構

### 3.1 DIKW金字塔理論

```
              ╱╲
             ╱  ╲  Wisdom (智慧)
            ╱────╲  ↓ 預測與決策支持
           ╱      ╲
          ╱  Know- ╲  Knowledge (知識)
         ╱   ledge  ╲  ↓ 模式識別與因果推理
        ╱────────────╲
       ╱              ╲
      ╱  Information  ╲  Information (資訊)
     ╱                 ╲  ↓ 統計分析與彙總
    ╱───────────────────╲
   ╱                     ╲
  ╱         Data         ╲  Data (數據)
 ╱                         ╲  原始測量值
╱───────────────────────────╲
```

**層級轉化特徵：**

| 層級 | 特徵 | 問題 | 技術 | Streamlit頁面 |
|------|------|------|------|---------------|
| **Data** | 原始、未處理 | "是什麼？" | 數據採集、存儲 | 數據總覽 |
| **Information** | 處理、有意義 | "有多少？" | 統計、聚合、可視化 | 統計分析 |
| **Knowledge** | 理解、有規律 | "為什麼？" | 分析、建模、推理 | 規律發現 |
| **Wisdom** | 應用、可行動 | "怎麼做？" | 預測、決策、建議 | 智慧決策、預測模型 |

### 3.2 Data層：原始數據

**定義：** 未經處理的原始測量值和觀察結果

**空氣質量數據範例：**
```
date: 2024-08-31 23:00
sitename: 中山
county: 台北市
aqi: 62
pm2.5: 17.0
o3: 35.0
windspeed: 2.3
```

**Streamlit實現：**
```python
st.header("原始數據查看")
st.dataframe(df.head(100))

# 數據品質檢查
st.subheader("數據品質")
missing_values = df.isnull().sum()
st.write(f"缺失值統計：")
st.dataframe(missing_values[missing_values > 0])
```

### 3.3 Information層：統計資訊

**定義：** 經過處理、彙總、有意義的數據

**空氣質量資訊範例：**

| 資訊類型 | 範例 | 生成方式 |
|---------|------|----------|
| **時間統計** | 台北市8月平均AQI為52（良好） | `df.groupby(['county', 'month'])['aqi'].mean()` |
| **空間比較** | 西部地區PM2.5比東部高30% | 區域分組統計 |
| **趨勢分析** | PM2.5濃度比去年同期下降15% | 年度對比計算 |
| **達標統計** | 本月空氣品質達標天數為25天 | `(df['aqi'] <= 100).sum()` |

**Streamlit實現：**
```python
st.header("統計資訊摘要")

# KPI指標卡片
col1, col2, col3, col4 = st.columns(4)
col1.metric("平均AQI", f"{df['aqi'].mean():.1f}", delta=f"{aqi_change:.1f}%")
col2.metric("監測站數", df['sitename'].nunique())
col3.metric("達標天數", (df['aqi'] <= 100).sum())
col4.metric("主要污染物", df['pollutant'].mode()[0])

# 統計圖表
st.subheader("月度趨勢")
monthly_avg = df.groupby('ym')['aqi'].mean()
st.line_chart(monthly_avg)
```

### 3.4 Knowledge層：知識與規律

**定義：** 從資訊中提煉出的模式、規律、因果關係

**知識範例：**

**範例1：季節性規律**
```
Information:
- 冬季平均PM2.5: 28.5 μg/m³
- 夏季平均PM2.5: 15.2 μg/m³

Knowledge:
→ 冬季PM2.5濃度顯著高於夏季（提高87%）
→ 原因：冬季大氣穩定、逆溫層形成、境外污染傳輸
→ 規律：每年11月-次年2月為污染高峰期
```

**範例2：風速與污染關聯**
```
Information:
- 風速<2m/s時，平均PM2.5: 32.1 μg/m³
- 風速>5m/s時，平均PM2.5: 12.3 μg/m³

Knowledge:
→ 風速與PM2.5濃度呈負相關（r=-0.65）
→ 當風速低於2m/s時，污染物不易擴散
→ 強風(>5m/s)可使PM2.5濃度降低62%
```

**Streamlit實現：**
```python
st.header("污染規律與洞察")

# 季節性分析
st.subheader("發現1：季節性污染模式")
seasonal_pattern = df.groupby('season')['pm2.5'].mean()
fig = px.bar(seasonal_pattern, title="各季節PM2.5平均濃度")
st.plotly_chart(fig)

st.info("""
**洞察：** 冬季PM2.5濃度比夏季高87%
- 原因：大氣穩定、逆溫層、境外傳輸
- 建議：冬季需加強污染管制措施
""")
```

### 3.5 Wisdom層：智慧與決策

**定義：** 基於知識的預測、建議、決策支持

**決策支持範例：**

```
Knowledge:
- 冬季PM2.5濃度高
- 風速低時污染難擴散
- 早晨7-9時為高峰

Wisdom (行動建議):
→ 預測：明日風速<2m/s，預測AQI將達120（不健康）
→ 建議：
  ✓ 敏感族群應避免戶外活動
  ✓ 學校考慮室內體育課
  ✓ 建議公眾搭乘大眾運輸
  ✓ 工廠考慮調整生產時段

→ 政策：
  ✓ 啟動空污應變措施
  ✓ 加強露天燃燒稽查
  ✓ 工業區排放減量10%
```

**智慧決策框架：**

| 情境 | 條件判斷 | 預測結果 | 行動建議 | 目標族群 |
|------|---------|---------|---------|---------|
| **健康警示** | AQI>100 | 敏感族群健康風險增加 | 減少戶外活動、配戴口罩 | 一般民眾 |
| **應變措施** | AQI>150 | 污染持續惡化 | 啟動應變機制、交通管制 | 政府單位 |
| **預防性措施** | 預測明日AQI>100 | 可能超標 | 提前通知、防護準備 | 學校、企業 |
| **污染管制** | 連續3天AQI>100 | 需加強管制 | 加強稽查、排放減量 | 環保部門 |

**Streamlit實現：**
```python
st.header("智慧決策與建議")

# 當前狀態評估
current_aqi = df['aqi'].iloc[-1]
aqi_level = get_aqi_level(current_aqi)

# 警示框
st.markdown(f"""
<div style='background-color: {alert_color[aqi_level]}; padding: 20px;'>
<h2>當前空氣品質：{aqi_level}</h2>
<h1>AQI: {current_aqi:.0f}</h1>
</div>
""", unsafe_allow_html=True)

# 預測模型
st.subheader("未來3天預測")
forecast = predict_aqi(df, days=3)
st.line_chart(forecast)

# 個人化建議
user_type = st.selectbox("您的身份",
    ["一般民眾", "敏感族群", "戶外工作者", "運動愛好者"])
advice = generate_advice(current_aqi, user_type)
st.info(advice)
```

---

## 4. Streamlit系統設計

### 4.1 系統架構總覽

**三層式架構：**
```
┌─────────────────────────────────────────────┐
│           前台 (Frontend)                    │
│  ┌──────────┬──────────────┬──────────────┐ │
│  │ Navbar   │   Sidebar    │    Canvas    │ │
│  │ (導航列)  │   (控制盤)    │   (主畫布)    │ │
│  └──────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────┘
                    ↓↑ 互動
┌─────────────────────────────────────────────┐
│          中間層 (Business Logic)             │
│  ┌──────────────────────────────────────┐  │
│  │  KDD流程 (Data → Info → Knowledge)   │  │
│  │  - 數據處理                          │  │
│  │  - 統計分析                          │  │
│  │  - 模型計算                          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                    ↓↑ 數據存取
┌─────────────────────────────────────────────┐
│           後台 (Backend/Data)                │
│  ┌──────────┬──────────────┬──────────────┐ │
│  │ Parquet  │   DuckDB     │  Data Loader │ │
│  │ (115MB)  │   (175MB)    │  (工具類)    │ │
│  └──────────┴──────────────┴──────────────┘ │
└─────────────────────────────────────────────┘
```

### 4.2 前台設計詳細規劃

#### 4.2.1 Navbar (導航列) 設計

**參照 AIp04空間與網站X.py 的導航設計**

```python
from streamlit_navigation_bar import st_navbar

# 導航列定義 - 對應DIKW層級
pages = [
    "[數據總覽]",       # Data層
    "[統計分析]",       # Information層
    "[規律發現]",       # Knowledge層
    "[智慧決策]",       # Wisdom層
    "[預測模型]"        # Wisdom層進階
]

page = st_navbar(pages)
```

**頁面對應關係：**

| 導航項 | DIKW層級 | 主要功能 | 關鍵指標 |
|--------|---------|---------|---------|
| 數據總覽 | Data | 原始數據查看、品質檢查 | 數據完整性、更新時間 |
| 統計分析 | Information | 統計圖表、趨勢分析 | AQI平均值、達標率 |
| 規律發現 | Knowledge | 模式識別、關聯分析 | 季節規律、影響因素 |
| 智慧決策 | Wisdom | 預測建議、決策支持 | 健康建議、政策建議 |
| 預測模型 | Wisdom | 時間序列預測、情境模擬 | 預測準確度、置信區間 |

#### 4.2.2 Sidebar (控制盤) 設計

**佈局結構：**
```
┌───────────────────────┐
│  ═══ 控制盤 ═══        │
├───────────────────────┤
│ 【數據篩選】           │
│  ├─ 時間範圍選擇       │
│  ├─ 縣市選擇          │
│  ├─ 站點選擇          │
│  └─ 污染物選擇        │
├───────────────────────┤
│ 【分析參數】           │
│  ├─ 統計方法          │
│  ├─ 聚合層級          │
│  └─ 可視化類型        │
├───────────────────────┤
│ 【當前狀態】           │
│  ├─ 數據筆數          │
│  ├─ 選擇摘要          │
│  └─ 計算狀態          │
├───────────────────────┤
│ 【操作LOG】            │
│  └─ 操作歷史記錄      │
└───────────────────────┘
```

**詳細控制項定義：**

```python
# 參照 AIp04/C2 輸入機制

st.sidebar.title("空氣質量分析控制盤")

# ===== 數據篩選區 =====
st.sidebar.header("【數據篩選】")

# 時間範圍選擇
date_range = st.sidebar.date_input(
    "選擇時間範圍",
    value=(datetime(2024, 1, 1), datetime(2024, 8, 31)),
    min_value=datetime(2016, 11, 25),
    max_value=datetime(2024, 8, 31)
)

# 縣市多選
counties = st.sidebar.multiselect(
    "選擇縣市",
    options=sorted(df['county'].unique()),
    default=['台北市', '新北市']
)

# 站點多選
stations = st.sidebar.multiselect(
    "選擇監測站",
    options=sorted(df[df['county'].isin(counties)]['sitename'].unique()),
    default=[]
)

# 污染物選擇
pollutants = st.sidebar.multiselect(
    "選擇污染物",
    options=['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2'],
    default=['PM2.5', 'PM10']
)

# ===== 分析參數區 =====
st.sidebar.header("【分析參數】")

# 統計方法
stat_method = st.sidebar.selectbox(
    "統計方法",
    ["平均值", "中位數", "最大值", "最小值", "總和"]
)

# 時間聚合層級
time_agg = st.sidebar.radio(
    "時間聚合",
    ["小時", "日", "週", "月", "季", "年"]
)

# AQI閾值設定
aqi_threshold = st.sidebar.slider(
    "AQI警示閾值",
    min_value=0,
    max_value=200,
    value=100,
    step=10
)

# ===== 當前狀態區 =====
st.sidebar.markdown("---")
st.sidebar.header("【當前狀態】")
st.sidebar.write(f"📊 數據筆數: {len(filtered_df):,}")
st.sidebar.write(f"📍 選擇站點: {len(stations) if stations else '全部'}")
st.sidebar.write(f"📅 時間範圍: {date_range[0]} ~ {date_range[1]}")

# ===== 操作日誌 =====
st.sidebar.markdown("---")
st.sidebar.header("【操作LOG】")
for i, log_entry in enumerate(st.session_state.log[-5:], 1):
    st.sidebar.text(f"{i}. {log_entry}")
```

#### 4.2.3 Canvas (主畫布) 設計

**頁面佈局模式：**

參照 **AIp04/C4 Streamlit的網頁佈局法**

**模式1：KPI指標卡片 (Columns)**
```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="平均AQI", value=f"{avg_aqi:.1f}", delta=f"{aqi_delta:.1f}%")
with col2:
    st.metric(label="監測站數", value=station_count)
with col3:
    st.metric(label="達標天數", value=good_days, delta=f"{good_days_delta}天")
with col4:
    st.metric(label="主要污染物", value=main_pollutant)
```

**模式2：Tabs切換 (用於同級內容)**
```python
tab1, tab2, tab3 = st.tabs(["時間趨勢", "空間分布", "污染物分析"])

with tab1:
    st.subheader("AQI時間序列趨勢")
    st.line_chart(time_series_data)

with tab2:
    st.subheader("各縣市污染地圖")
    st.map(location_data)

with tab3:
    st.subheader("污染物關聯矩陣")
    st.plotly_chart(correlation_heatmap)
```

**模式3：Expander展開式 (詳細資訊)**
```python
with st.expander("詳細數據表格"):
    st.dataframe(detailed_data)

with st.expander("統計摘要"):
    st.write(statistical_summary)
```

### 4.3 頁面功能設計概要

#### 頁面1：數據總覽（Data層）
- 原始數據表格展示
- 數據品質檢查（缺失值、異常值）
- 數據基本統計摘要
- 數據下載功能

#### 頁面2：統計分析（Information層）
- KPI指標卡片（平均AQI、達標率等）
- 時間序列趨勢圖
- Crosstab交叉表分析
- 分布直方圖

#### 頁面3：規律發現（Knowledge層）
- 季節性模式分析
- 地理分布規律
- 氣象影響分析（風速、風向）
- 污染物相關性分析

#### 頁面4：智慧決策（Wisdom層）
- 當前空氣品質警示
- 未來3天AQI預測
- 個人化健康建議
- 政策應變建議

#### 頁面5：預測模型（Wisdom層進階）
- 多種預測模型選擇
- 預測結果可視化
- 模型性能評估
- 情境模擬（調整影響因素）

### 4.4 Session State管理

**參照 AIp04空間與網站X.py 的 session_state 管理**

```python
def init_session_state():
    """初始化所有session變量"""
    sss = st.session_state

    # 數據相關
    if 'df' not in sss:
        sss.df = None

    # 使用者選擇
    if 'selected_counties' not in sss:
        sss.selected_counties = []
    if 'selected_stations' not in sss:
        sss.selected_stations = []
    if 'date_range' not in sss:
        sss.date_range = None

    # 分析結果
    if 'analysis_result' not in sss:
        sss.analysis_result = None

    # 操作日誌
    if 'log' not in sss:
        sss.log = ["系統初始化"]

    # 使用者資訊
    if 'username' not in sss:
        sss.username = ""

    return sss

# 日誌記錄函數
def add_log(message):
    """添加操作日誌"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.log.append(f"[{timestamp}] {message}")
```

---

## 5. 技術實現細節

### 5.1 數據載入模組

**使用已建立的 data_loader.py：**

```python
from src.main.python.utils.data_loader import (
    AirQualityDataLoader,
    load_air_quality_data,
    query_air_quality
)

# 方式1: 使用便捷函數
@st.cache_data
def load_data(start_date, end_date, counties=None):
    """載入數據並快取"""
    df = load_air_quality_data(
        start_date=start_date,
        end_date=end_date,
        county=counties[0] if counties else None
    )
    return df

# 方式2: 使用SQL查詢（更高效）
@st.cache_data
def query_data_by_sql(start_date, end_date, counties):
    """使用SQL查詢數據"""
    counties_str = "', '".join(counties)
    sql = f"""
        SELECT *
        FROM air_quality
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
          AND county IN ('{counties_str}')
    """
    df = query_air_quality(sql)
    return df

# 方式3: 使用DataLoader類（最靈活）
@st.cache_resource
def get_data_loader():
    """獲取單例DataLoader"""
    return AirQualityDataLoader()
```

### 5.2 數據處理與轉換

**參照 AIp04 的 KDD3 數據轉換：**

```python
def prepare_data(df):
    """數據準備和標籤生成"""
    # (1) 日期相關標籤
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek

    # (2) 連續數值的離散化
    df['quarter'] = pd.cut(df['month'], bins=[0,3,6,9,12],
                           labels=['Q1','Q2','Q3','Q4'])
    df['season'] = pd.cut(df['month'], bins=[0,3,6,9,12],
                          labels=['冬季','春季','夏季','秋季'])
    df['aqi_level'] = pd.cut(df['aqi'], bins=[0,50,100,150,200,300,500],
                             labels=['良好','普通','對敏感族群不健康',
                                    '不健康','非常不健康','危害'])
    df['wind_level'] = pd.cut(df['windspeed'], bins=[0,1,3,5,float('inf')],
                              labels=['無風','微風','輕風','強風'])

    # (3) 文字處理的標籤
    df['yq'] = df['year'].astype(str).str[-2:] + 'Q' + df['quarter'].astype(str)
    df['ym'] = df['date'].dt.to_period('M').astype(str)

    # (4) 布林標籤
    df['is_weekend'] = df['dayofweek'] >= 5
    df['is_exceed'] = df['aqi'] > 100

    return df
```

### 5.3 可視化函數庫

```python
import plotly.express as px
import plotly.graph_objects as go

def create_time_series_plot(df, y_column='aqi', title=None):
    """創建時間序列圖"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df[y_column],
                            mode='lines', name=y_column.upper()))

    if y_column == 'aqi':
        fig.add_hline(y=100, line_dash="dash", annotation_text="普通上限(100)")
        fig.add_hline(y=150, line_dash="dash", annotation_text="不健康(150)")

    fig.update_layout(title=title or f"{y_column.upper()} 時間序列",
                     xaxis_title="日期", yaxis_title=y_column.upper())
    return fig

def create_heatmap(df, x_col, y_col, value_col, agg_func='mean'):
    """創建熱力圖"""
    pivot_data = df.pivot_table(index=y_col, columns=x_col,
                                values=value_col, aggfunc=agg_func)

    fig = go.Figure(data=go.Heatmap(z=pivot_data.values,
                                     x=pivot_data.columns,
                                     y=pivot_data.index,
                                     colorscale='RdYlGn_r'))
    return fig

def create_map_plot(df):
    """創建地理分布地圖"""
    station_data = df.groupby(['sitename', 'latitude', 'longitude']).agg({
        'aqi': 'mean', 'pm2.5': 'mean'
    }).reset_index()

    fig = px.scatter_mapbox(station_data, lat='latitude', lon='longitude',
                           size='pm2.5', color='aqi',
                           hover_name='sitename',
                           color_continuous_scale='RdYlGn_r',
                           zoom=7, center={"lat": 23.5, "lon": 121})
    fig.update_layout(mapbox_style="open-street-map")
    return fig
```

### 5.4 預測模型實現

```python
def predict_aqi_simple_ma(df, days=3, window=7):
    """簡單移動平均預測"""
    df_sorted = df.sort_values('date')
    ma = df_sorted['aqi'].rolling(window=window).mean()

    last_date = df_sorted['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=days, freq='D')

    forecast_value = ma.iloc[-1]
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': [forecast_value] * days,
        'confidence_lower': [forecast_value * 0.9] * days,
        'confidence_upper': [forecast_value * 1.1] * days
    })
    return forecast_df

def generate_personalized_advice(aqi, user_group):
    """生成個人化健康建議"""
    advice_matrix = {
        '一般民眾': {
            (0, 50): "空氣品質良好，適合各種戶外活動。",
            (51, 100): "空氣品質普通，可正常戶外活動。",
            (101, 150): "建議減少長時間劇烈運動。",
            (151, 200): "應減少戶外活動，外出時配戴口罩。",
            (201, 500): "避免戶外活動，關閉門窗。"
        },
        '敏感族群': {
            (0, 50): "空氣品質良好，可正常活動。",
            (51, 100): "空氣品質普通，注意身體狀況。",
            (101, 150): "應減少戶外活動，必要時配戴口罩。",
            (151, 500): "避免戶外活動，留在室內。"
        }
    }

    for (min_aqi, max_aqi), advice in advice_matrix[user_group].items():
        if min_aqi <= aqi <= max_aqi:
            return advice
    return "數據異常，請查看最新官方公告。"
```

---

## 6. 開發路線圖

### 6.1 開發階段規劃

**Phase 1: 基礎架構（已完成）✅**
- ✅ CSV to Parquet 轉換
- ✅ DuckDB 數據庫建立
- ✅ Data Loader 工具類
- ✅ 數據驗證

**Phase 2: Streamlit基礎框架（Week 1）**
- 🎯 建立主程式結構
- 🎯 實現導航系統
- 🎯 完成Sidebar控制盤
- 🎯 實現Session State管理

**Phase 3: Data & Information 層（Week 2）**
- 🎯 頁面1：數據總覽
- 🎯 頁面2：統計分析
- 🎯 基本可視化功能
- 🎯 交叉表分析

**Phase 4: Knowledge & Wisdom 層（Week 3）**
- 🎯 頁面3：規律發現
- 🎯 頁面4：智慧決策
- 🎯 相關性分析
- 🎯 健康建議系統

**Phase 5: 預測模型（Week 4）**
- 🎯 頁面5：預測模型
- 🎯 時間序列預測
- 🎯 情境模擬功能
- 🎯 模型性能評估

**Phase 6: 優化與部署（Week 5）**
- 🎯 性能優化
- 🎯 UI/UX改進
- 🎯 文檔完善
- 🎯 部署到雲端

### 6.2 文件產出時程

| 產出物 | 完成時間 | 內容 |
|--------|---------|------|
| **5頁PPT** | Week 1 | 數據介紹、維度分析、DIKW架構、系統設計、成果展示 |
| **Streamlit Demo** | Week 3 | Data & Information層完整功能 |
| **完整系統** | Week 5 | 全部5個頁面，所有DIKW層級實現 |
| **技術文檔** | Week 5 | 系統說明、API文檔、部署指南 |

### 6.3 技術棧

**後端技術：**
- Python 3.12+
- Pandas 2.3+
- DuckDB 1.4+
- PyArrow 21+
- NumPy

**前端技術：**
- Streamlit 1.28+
- Plotly Express
- streamlit-navigation-bar

**數據科學：**
- Scikit-learn
- Prophet (時間序列，可選)

**部署：**
- Streamlit Cloud
- Docker (可選)

---

## 總結

本規劃文檔完整定義了：

1. ✅ **數據結構** - 5.88M筆記錄，25個欄位，8年歷史數據
2. ✅ **維度分析** - SPCT四維模型（空間、污染物、條件、時間）
3. ✅ **DIKW架構** - 從數據到智慧的四層轉化路徑
4. ✅ **系統設計** - Streamlit三層架構，5個主要頁面
5. ✅ **技術細節** - 完整的代碼範例和實現方案
6. ✅ **開發計畫** - 5週開發路線圖

**下一步行動：**
1. 根據此規劃建立Streamlit應用主程式
2. 逐步實現各個頁面功能
3. 準備5頁PPT簡報
4. 測試與優化系統性能

---

**文檔版本：** 1.0
**最後更新：** 2025-10-13
**維護者：** Claude Code
**狀態：** ✅ 規劃完成，準備進入開發階段
