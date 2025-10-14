# Air Quality Data Analysis

Taiwan air quality monitoring data analysis repository with optimized data formats for efficient processing and analysis.

## Overview

This repository contains ~5.9 million records of Taiwan air quality monitoring data spanning from 2016 to 2024, collected from 123 monitoring stations across the country. The data has been converted from CSV to optimized formats (Parquet and DuckDB) for efficient storage and fast queries.

## Data Formats

### Original Format
- **CSV File**: 
  [from Kaggle](https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024)
  - `air_quality.csv` (801 MB)
- 5,882,209 records
- 25 columns including timestamps, pollutant measurements, and geolocation

### Optimized Formats

#### 1. Parquet Format (Recommended for Python)
- **Location**: `data/processed/`
- **Size**: ~150-200 MB (70-80% compression)
- **Features**:
  - Columnar storage for fast analytics
  - Efficient compression
  - Year-based partitioning
  - Native pandas/polars support
  - 10-100x faster than CSV

#### 2. DuckDB Database (Recommended for SQL)
- **Location**: `data/processed/air_quality.duckdb`
- **Features**:
  - Full SQL query support
  - Analytical query optimization
  - Pre-built views for common patterns
  - Zero-copy Parquet integration
  - Excellent performance for complex queries

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Note:
- Use Streamlit >= 1.39.0 to avoid deprecated keyword warnings for Plotly and to support `width='stretch'`.
- If you already have Streamlit installed, upgrade with:

```bash
pip install -U streamlit
```

Required packages:
- pandas >= 2.0.0
- pyarrow >= 12.0.0
- duckdb >= 0.9.0
- numpy >= 1.24.0

### 2. Convert CSV to Optimized Formats

If you haven't converted the data yet:

```bash
# Convert CSV to Parquet (with year partitioning)
python3 src/main/python/core/csv_to_parquet_converter.py

# Create DuckDB database from Parquet files
python3 src/main/python/core/duckdb_database_creator.py

# Validate conversion integrity
python3 src/main/python/core/data_validator.py
```

### 3. Load and Analyze Data

#### Option A: Using Python (Parquet)

```python
from src.main.python.utils.data_loader import AirQualityDataLoader

# Initialize loader
loader = AirQualityDataLoader()

# Load data with filters
df = loader.load_parquet(
    start_date='2024-08-01',
    end_date='2024-08-31',
    counties=['Taipei City']
)

# Or load by year
df_2024 = loader.load_by_year(2024)
```

#### Option B: Using SQL (DuckDB)

```python
from src.main.python.utils.data_loader import AirQualityDataLoader

loader = AirQualityDataLoader()

# Execute SQL query
result = loader.query_db("""
    SELECT
        county,
        AVG(aqi) as avg_aqi,
        AVG(pm2_5) as avg_pm25
    FROM air_quality
    WHERE year = 2024
    GROUP BY county
    ORDER BY avg_aqi DESC
""")
```

#### Option C: Quick Functions

```python
from src.main.python.utils.data_loader import load_air_quality_data, query_air_quality

# Quick load with filters
df = load_air_quality_data(
    start_date='2024-08-01',
    end_date='2024-08-31',
    county='Taipei City'
)

# Quick SQL query
result = query_air_quality("""
    SELECT sitename, AVG(aqi) as avg_aqi
    FROM air_quality
    WHERE year = 2024
    GROUP BY sitename
    ORDER BY avg_aqi DESC
    LIMIT 10
""")
```

## Streamlit App

### Run the App

```bash
streamlit run src/main/python/app.py
```

- Default port is configured to `9000` in `.streamlit/config.toml`.
- Ensure optimized data exists under `data/processed/` (Parquet or DuckDB).

### Control Panel Guide（側邊欄控制盤）

- 【數據載入】
  - 顯示可用日期範圍；若資料不存在會提示檢查路徑
- 【數據篩選】
  - 時間範圍：選擇開始與結束日期（必選）
  - 縣市：可多選；留空表示全部
  - 監測站：會依選擇的縣市篩出站點；留空表示全部
  - 主要污染物：僅篩選主污染物欄位
- 【分析參數】
  - 統計方法：平均值/中位數/最大值/最小值
  - 時間聚合層級：小時/日/週/月/季/年
  - AQI 警示閾值：用於趨勢線與提醒
- 「🔄 載入/更新數據」
  - 依上述條件載入資料並套用
- 【當前狀態】
  - 顯示已載入的筆數、站數、資料時間範圍與平均 AQI
- 【操作LOG日誌】
  - 顯示最近操作與提示

提示：
- 縣市/站點留空 = 不限制（全選）
- 重新調整條件後，需再次按下「載入/更新數據」才會生效
- 本版本已移除側邊欄的「用戶輸入/提交建議」功能（不再需要）

### Troubleshooting: Scroll Lock / 無法滾動

若右側主視窗無法捲動，可能是第三方導航列樣式影響。可使用以下快速解法：

1) 預設已採用側邊欄導航（不載入頂部導航列），通常可避免此問題。

2) 若仍想使用頂部導航列，可手動啟用：

```bash
USE_TOP_NAV=1 streamlit run src/main/python/app.py
```

Windows PowerShell:

```powershell
$env:USE_TOP_NAV=1; streamlit run src/main/python/app.py
```

若啟用後再次遇到捲動問題，請改回預設不啟用頂部導航。強化 CSS 已內建，重新載入亦可嘗試。

### Navigation（DIKW 模型）

- 數據總覽：原始數據與基本指標
- 統計分析：分布、統計量、趨勢
- 規律發現：相關性、熱力圖、風玫瑰圖
- 智慧決策：健康建議與策略參考
- 預測模型：預測結果展示（頁面架構已備妥）

## Data Schema

### Columns

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Measurement timestamp (hourly) |
| sitename | category | Monitoring station name |
| county | category | Administrative region |
| aqi | float32 | Air Quality Index |
| pollutant | category | Primary pollutant (PM2.5, O3, etc.) |
| status | category | Air quality status (Good, Moderate, etc.) |
| so2 | float32 | Sulfur dioxide (ppb) |
| co | float32 | Carbon monoxide (ppm) |
| o3 | float32 | Ozone (ppb) |
| o3_8hr | float32 | 8-hour ozone average |
| pm10 | float32 | PM10 particles (μg/m³) |
| pm2.5 | float32 | PM2.5 particles (μg/m³) |
| no2 | float32 | Nitrogen dioxide (ppb) |
| nox | float32 | Nitrogen oxides (ppb) |
| no | float32 | Nitric oxide (ppb) |
| windspeed | float32 | Wind speed (m/s) |
| winddirec | float32 | Wind direction (degrees) |
| co_8hr | float32 | 8-hour CO average |
| pm2.5_avg | float32 | PM2.5 running average |
| pm10_avg | float32 | PM10 running average |
| so2_avg | float32 | SO2 running average |
| longitude | float64 | Station longitude |
| latitude | float64 | Station latitude |
| siteid | float32 | Unique station identifier |
| year | int | Year (partition column) |

## Pre-built Database Views

The DuckDB database includes helpful views:

### 1. `daily_averages`
Daily aggregated metrics by station
```sql
SELECT * FROM daily_averages
WHERE date >= '2024-08-01' AND county = 'Taipei City'
```

### 2. `monthly_summary`
Monthly statistics by county
```sql
SELECT * FROM monthly_summary
WHERE year = 2024 AND county = 'Taipei City'
ORDER BY month
```

### 3. `high_pollution_events`
Records with AQI > 100
```sql
SELECT * FROM high_pollution_events
WHERE year = 2024
ORDER BY aqi DESC
LIMIT 20
```

### 4. `station_metadata`
Station information and coverage
```sql
SELECT * FROM station_metadata
ORDER BY sitename
```

## Performance Comparison

Based on benchmarks with 100,000 rows:

| Operation | CSV | Parquet | DuckDB | Speedup |
|-----------|-----|---------|--------|---------|
| File Size | 801 MB | ~160 MB | ~180 MB | 5x smaller |
| Load Time | ~8s | ~0.8s | ~0.5s | 10-16x faster |
| Query Time | ~10s | ~2s | ~0.2s | 10-50x faster |

## Project Structure

```
AirQuality/
├── CLAUDE.md                   # Project guidelines for Claude Code
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── air_quality.csv             # Original dataset (801 MB)
├── data/
│   ├── processed/              # Converted Parquet files (year-partitioned)
│   │   ├── year=2016/
│   │   ├── year=2017/
│   │   ├── ...
│   │   └── year=2024/
│   └── temp/                   # Temporary processing files
├── src/
│   └── main/
│       └── python/
│           ├── core/           # Core business logic
│           │   ├── csv_to_parquet_converter.py
│           │   ├── duckdb_database_creator.py
│           │   ├── data_validator.py
│           │   └── performance_benchmark.py
│           └── utils/          # Utility functions
│               └── data_loader.py
├── notebooks/
│   └── exploratory/
│       └── data_format_usage_examples.ipynb
└── tests/
    └── unit/                   # Unit tests (to be added)
```

## Usage Examples

For comprehensive examples, see the Jupyter notebook:
- `notebooks/exploratory/data_format_usage_examples.ipynb`

The notebook demonstrates:
- Loading data with filters
- SQL queries with DuckDB
- Time series analysis
- Spatial analysis
- Pollution event detection
- Performance comparisons

## Common Use Cases

### 1. Time Series Analysis
```python
loader = AirQualityDataLoader()
df = loader.load_parquet(
    start_date='2024-01-01',
    end_date='2024-12-31',
    counties=['Taipei City']
)

# Calculate daily averages
daily = df.groupby(df['date'].dt.date)['aqi'].mean()
```

### 2. Spatial Analysis
```python
# Get average pollution by station with coordinates
spatial_data = loader.query_db("""
    SELECT
        sitename,
        AVG(longitude) as lon,
        AVG(latitude) as lat,
        AVG(aqi) as avg_aqi
    FROM air_quality
    WHERE year = 2024
    GROUP BY sitename
""")
```

### 3. High Pollution Events
```python
# Find pollution events above threshold
events = loader.query_db("""
    SELECT * FROM high_pollution_events
    WHERE year = 2024 AND aqi > 150
    ORDER BY aqi DESC
""")
```

## Data Quality Notes

- **Missing Values**: Some pollutant measurements may be null
- **Temporal Coverage**: Hourly measurements with potential gaps
- **Geographic Coverage**: 123 stations across Taiwan
- **Data Period**: 2016-11-25 to 2024-08-31

## Performance Tips

1. **Use Parquet for column-specific queries** - Only load columns you need
2. **Use DuckDB for complex analytics** - Leverage SQL optimization
3. **Filter by date range** - Take advantage of year partitioning
4. **Use pre-built views** - Common patterns are already optimized
5. **Load by year** - More efficient than full dataset for recent analysis

## Validation

To verify data integrity after conversion:

```bash
python3 src/main/python/core/data_validator.py
```

This checks:
- Row count consistency
- Schema validation
- Value range verification
- Missing data analysis
- Statistical comparison

## Benchmark

To run performance benchmarks:

```bash
python3 src/main/python/core/performance_benchmark.py
```

This measures:
- File sizes and compression ratios
- Load time performance
- Query performance
- Format comparisons
