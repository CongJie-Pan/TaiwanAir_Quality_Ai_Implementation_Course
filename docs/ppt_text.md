# 台灣空氣質量數據分析系統 — 簡報文案（5頁精簡版）

— 基於 PLANNING.md × DIKW × AIP04/SPCT —

---

## 第1頁｜有哪些數據（Data Overview）
- 來源與範圍：環保署監測網；2016-11-25～2024-08-31（8年）
- 規模：5,882,208 筆；123 監測站；22 縣市；每小時紀錄
- 主要欄位：
  - 時間（T）：`date`
  - 空間（S）：`sitename`, `county`, `longitude`, `latitude`, `siteid`
  - 污染物（P）：`pm2.5`, `pm10`, `o3`, `o3_8hr`, `co`, `co_8hr`, `so2`, `so2_avg`, `no2`, `no`, `nox`
  - 指標/氣象：`aqi`, `windspeed`, `winddirec`
- 效能優化：CSV 801MB → Parquet 115MB（壓縮 85.7%）；DuckDB 查詢提速 20–50x

---

## 第2頁｜數據維度分析（AIP04／SPCT）— 概念
- SPCT 四維：S（空間）× P（污染物）× C（條件）× T（時間）
- S：區域 → 縣市 → 監測站（例：北/中/南/東/離島 → 台北市 → 中山站）
- P：懸浮微粒（PM2.5/PM10）、氣態（O3/CO/SO2/NOx）、綜合（AQI）
- T：年/季/月份/週/日/時（支持年季`yq`、年月`ym`等標籤）
- 典型分析視角：
  - 空間比較（縣市/站點的 AQI/PM2.5 差異）
  - 時序趨勢（`ym` 月均AQI、年度對比）
  - 污染物結構（不同 P 的占比/關聯）

---

## 第3頁｜數據維度分析（AIP04／SPCT）— 條件與衍生
- C（條件）分層：
  - 季節：春/夏/秋/冬（以月份產生 `season`）
  - 風速：無風/微風/輕風/強風（以 `windspeed` 區間離散）
  - 時段：早晨/上午/下午/傍晚/夜間（以小時分段）
  - 其他：是否週末（`is_weekend`）等
- 衍生標籤：`season`, `wind_level`, `time_period`, `yq`, `ym`, `aqi_level`
- 常用統計：平均/中位、達標率（AQI≤100）、Top-N 站點、環比/同比
- 常用圖表：Crosstab 熱力圖（S×T）、折線（T趨勢）、長條（P分佈）

---

## 第4頁｜該如何呈現在系統上（System Design）
- 架構：Navbar + Sidebar + 主畫布（Streamlit）
- 資料層：Parquet + DuckDB（快速查詢、低資源占用）
- 互動控制（Sidebar）：時間範圍、縣市/站點、多污染物、聚合層級
- 頁面配置（建議5頁）：
  - 總覽（Data）：原始表格、品質檢查、KPI 概覽
  - 統計（Information）：月均趨勢、達標率、空間比較
  - 規律（Knowledge）：季節性/風速影響、關聯分析
  - 決策（Wisdom）：告警規則、健康建議、行動清單
  - 地圖（Spatial）：站點分布與 AQI 著色、點選細節

---

## 第5頁｜展示範例與體驗指標（Examples & UX）
- 視覺範例：
  - 月均 AQI 趨勢（折線）／區域對比（群組長條）
  - S×T 熱力（Crosstab）／PM2.5 與風速散點（關聯）
  - 地圖圈層（站點顏色= AQI 等級）
- 互動體驗：即時篩選、多選維度、懸浮提示、下載圖表
- 體驗指標：載入 < 2s｜查詢 < 1s｜首屏渲染 < 3s
- 產出交付：Streamlit 站點、PPT（本稿）、技術文件
