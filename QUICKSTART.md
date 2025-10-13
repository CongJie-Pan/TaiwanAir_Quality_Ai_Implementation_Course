# Quick Start Guide

Get started with optimized air quality data formats in 5 minutes.

## Prerequisites

```bash
# Install required packages
pip install pandas pyarrow duckdb numpy matplotlib seaborn jupyter
```

## Step 1: Convert Data (One-Time Setup)

Run the conversion pipeline to create optimized formats:

```bash
python3 run_conversion.py
```

This will:
- Convert CSV to Parquet (~5-10 minutes for 801MB file)
- Create DuckDB database
- Validate data integrity
- Run performance benchmarks

**Note**: You only need to do this once. The converted files will be saved in `data/processed/`.

## Step 2: Load and Analyze

### Quick Example

```python
from src.main.python.utils.data_loader import load_air_quality_data

# Load one month of data for Taipei City
df = load_air_quality_data(
    start_date='2024-08-01',
    end_date='2024-08-31',
    county='Taipei City'
)

print(f"Loaded {len(df):,} rows")
print(df[['date', 'sitename', 'aqi', 'pm2.5']].head())
```

### SQL Query Example

```python
from src.main.python.utils.data_loader import query_air_quality

# Find top 10 most polluted stations in 2024
result = query_air_quality("""
    SELECT
        sitename,
        county,
        AVG(aqi) as avg_aqi,
        AVG(pm2_5) as avg_pm25
    FROM air_quality
    WHERE year = 2024
    GROUP BY sitename, county
    ORDER BY avg_aqi DESC
    LIMIT 10
""")

print(result)
```

## Step 3: Explore Examples

Open the Jupyter notebook for comprehensive examples:

```bash
jupyter notebook notebooks/exploratory/data_format_usage_examples.ipynb
```

## Common Tasks

### Load Specific Year
```python
from src.main.python.utils.data_loader import AirQualityDataLoader

loader = AirQualityDataLoader()
df_2024 = loader.load_by_year(2024)
```

### Get Station List
```python
stations = loader.get_station_list()
print(stations[['sitename', 'county']].head(10))
```

### Get Summary Statistics
```python
stats = loader.get_summary_stats(county='Taipei City')
print(stats)
```

### Time Series Analysis
```python
# Load data
df = loader.load_parquet(
    start_date='2024-08-01',
    end_date='2024-08-31',
    counties=['Taipei City']
)

# Calculate daily averages
daily = df.groupby(df['date'].dt.date)['aqi'].mean()

# Plot
import matplotlib.pyplot as plt
daily.plot(title='Daily Average AQI - Taipei City')
plt.show()
```

## Performance Benefits

| Operation | CSV | Optimized | Speedup |
|-----------|-----|-----------|---------|
| File Size | 801 MB | ~160 MB | 5x smaller |
| Load Time | ~8s | ~0.5s | 16x faster |
| Query Time | ~10s | ~0.2s | 50x faster |

## Troubleshooting

### Module Import Errors
If you get import errors, make sure you're running from the project root:
```bash
cd /path/to/AirQuality
python3 your_script.py
```

Or add the project to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src' / 'main' / 'python'))
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data Not Found
Make sure you've run the conversion first:
```bash
python3 run_conversion.py
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [Jupyter notebook](notebooks/exploratory/data_format_usage_examples.ipynb)
- Check [CLAUDE.md](CLAUDE.md) for development guidelines
- Review the [source code](src/main/python/) for implementation details

## Support

For issues or questions:
1. Check this guide and README.md
2. Review the example notebook
3. Examine the source code
4. Check error logs for details

---

**Happy Analyzing! 🎉**
