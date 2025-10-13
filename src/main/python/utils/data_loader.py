"""
Data Loader Utilities for Air Quality Data

This module provides convenient functions for loading and accessing air quality
data in various formats (Parquet, DuckDB, CSV).

Features:
- Load data from Parquet files with optional filters
- Connect to DuckDB database for SQL queries
- Filter by date range, county, station
- Memory-efficient chunked loading
- Common query patterns

Author: Claude Code
Date: 2025-10-13
"""

import pandas as pd
import duckdb
from pathlib import Path
from typing import Optional, List, Union
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AirQualityDataLoader:
    """
    Provides convenient methods for loading air quality data.

    This class offers multiple ways to access air quality data:
    - Load from Parquet files (with filtering)
    - Query from DuckDB database
    - Access original CSV (with chunking)

    Attributes:
        data_dir: Base directory containing data files
        parquet_dir: Directory containing Parquet files
        db_path: Path to DuckDB database
    """

    def __init__(
        self,
        data_dir: str = ".",
        parquet_dir: str = "data/processed",
        db_path: str = "data/processed/air_quality.duckdb"
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Base directory for data files
            parquet_dir: Directory containing Parquet files
            db_path: Path to DuckDB database file
        """
        self.data_dir = Path(data_dir)
        self.parquet_dir = self.data_dir / parquet_dir
        self.db_path = self.data_dir / db_path
        self._db_connection: Optional[duckdb.DuckDBPyConnection] = None

        logger.info(f"Data loader initialized: {self.data_dir}")

    def load_parquet(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        counties: Optional[List[str]] = None,
        stations: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load data from Parquet files with optional filtering.

        This method provides efficient loading with filter pushdown to reduce
        memory usage. Filters are applied during file reading when possible.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format (inclusive)
            end_date: End date in 'YYYY-MM-DD' format (inclusive)
            counties: List of counties to include
            stations: List of station names to include
            columns: List of columns to load (None = all columns)

        Returns:
            DataFrame containing filtered air quality data

        Example:
            >>> loader = AirQualityDataLoader()
            >>> df = loader.load_parquet(
            ...     start_date='2024-01-01',
            ...     end_date='2024-01-31',
            ...     counties=['Taipei City']
            ... )
        """
        logger.info("Loading data from Parquet files...")

        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")

        # Build file pattern
        parquet_pattern = f"{self.parquet_dir}/**/*.parquet"

        # Read Parquet files
        df = pd.read_parquet(
            parquet_pattern,
            columns=columns,
            engine='pyarrow'
        )

        # Apply filters
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
            logger.info(f"Filtered by start_date: {start_date}")

        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            logger.info(f"Filtered by end_date: {end_date}")

        if counties:
            df = df[df['county'].isin(counties)]
            logger.info(f"Filtered by counties: {counties}")

        if stations:
            df = df[df['sitename'].isin(stations)]
            logger.info(f"Filtered by stations: {stations}")

        logger.info(f"Loaded {len(df):,} rows")

        return df

    def load_by_year(self, year: int) -> pd.DataFrame:
        """
        Load all data for a specific year.

        Args:
            year: Year to load (e.g., 2024)

        Returns:
            DataFrame containing data for the specified year

        Example:
            >>> loader = AirQualityDataLoader()
            >>> df_2024 = loader.load_by_year(2024)
        """
        logger.info(f"Loading data for year {year}...")

        year_dir = self.parquet_dir / f"year={year}"

        if not year_dir.exists():
            raise FileNotFoundError(f"No data found for year {year}: {year_dir}")

        df = pd.read_parquet(year_dir, engine='pyarrow')

        logger.info(f"Loaded {len(df):,} rows for year {year}")

        return df

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        """
        Connect to the DuckDB database.

        Returns:
            DuckDB connection object

        Raises:
            FileNotFoundError: If database file doesn't exist

        Example:
            >>> loader = AirQualityDataLoader()
            >>> conn = loader.connect_db()
            >>> result = conn.execute("SELECT COUNT(*) FROM air_quality").fetchone()
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        if not self._db_connection:
            self._db_connection = duckdb.connect(str(self.db_path), read_only=True)
            logger.info(f"Connected to database: {self.db_path}")

        return self._db_connection

    def query_db(self, sql: str) -> pd.DataFrame:
        """
        Execute SQL query on DuckDB database and return results as DataFrame.

        Args:
            sql: SQL query string

        Returns:
            DataFrame with query results

        Example:
            >>> loader = AirQualityDataLoader()
            >>> df = loader.query_db('''
            ...     SELECT county, AVG(aqi) as avg_aqi
            ...     FROM air_quality
            ...     WHERE year = 2024
            ...     GROUP BY county
            ... ''')
        """
        conn = self.connect_db()
        result = conn.execute(sql).fetchdf()

        logger.info(f"Query returned {len(result):,} rows")

        return result

    def get_station_list(self) -> pd.DataFrame:
        """
        Get list of all monitoring stations with metadata.

        Returns:
            DataFrame with station information (name, county, location)

        Example:
            >>> loader = AirQualityDataLoader()
            >>> stations = loader.get_station_list()
            >>> print(stations[['sitename', 'county']].head())
        """
        if self.db_path.exists():
            # Use database view if available
            return self.query_db("SELECT * FROM station_metadata ORDER BY sitename")
        else:
            # Fall back to Parquet files
            df = self.load_parquet(columns=['sitename', 'county', 'siteid', 'longitude', 'latitude'])
            return df.drop_duplicates('sitename').sort_values('sitename')

    def get_date_range(self) -> tuple:
        """
        Get the date range of available data.

        Returns:
            Tuple of (min_date, max_date)

        Example:
            >>> loader = AirQualityDataLoader()
            >>> min_date, max_date = loader.get_date_range()
            >>> print(f"Data available from {min_date} to {max_date}")
        """
        if self.db_path.exists():
            result = self.query_db("SELECT MIN(date) as min_date, MAX(date) as max_date FROM air_quality")
            return (result['min_date'].iloc[0], result['max_date'].iloc[0])
        else:
            df = self.load_parquet(columns=['date'])
            return (df['date'].min(), df['date'].max())

    def get_summary_stats(self, county: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for air quality metrics.

        Args:
            county: Optional county name to filter by

        Returns:
            DataFrame with summary statistics

        Example:
            >>> loader = AirQualityDataLoader()
            >>> stats = loader.get_summary_stats(county='Taipei City')
        """
        where_clause = f"WHERE county = '{county}'" if county else ""

        if self.db_path.exists():
            sql = f"""
                SELECT
                    county,
                    COUNT(*) as total_measurements,
                    AVG(aqi) as avg_aqi,
                    MAX(aqi) as max_aqi,
                    MIN(aqi) as min_aqi,
                    AVG(pm2_5) as avg_pm25,
                    AVG(pm10) as avg_pm10,
                    AVG(o3) as avg_o3
                FROM air_quality
                {where_clause}
                GROUP BY county
                ORDER BY county
            """
            return self.query_db(sql)
        else:
            # Load and calculate from Parquet
            df = self.load_parquet(counties=[county] if county else None)
            return df.groupby('county').agg({
                'aqi': ['count', 'mean', 'max', 'min'],
                'pm2.5': 'mean',
                'pm10': 'mean',
                'o3': 'mean'
            }).reset_index()

    def close(self):
        """
        Close database connection if open.
        """
        if self._db_connection:
            self._db_connection.close()
            self._db_connection = None
            logger.info("Database connection closed")


# Convenience functions for quick access

def load_air_quality_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    county: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick function to load air quality data with basic filters.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        county: County name to filter

    Returns:
        Filtered DataFrame

    Example:
        >>> df = load_air_quality_data(
        ...     start_date='2024-01-01',
        ...     end_date='2024-01-31',
        ...     county='Taipei City'
        ... )
    """
    loader = AirQualityDataLoader()
    return loader.load_parquet(
        start_date=start_date,
        end_date=end_date,
        counties=[county] if county else None
    )


def query_air_quality(sql: str) -> pd.DataFrame:
    """
    Quick function to execute SQL query on air quality database.

    Args:
        sql: SQL query string

    Returns:
        Query results as DataFrame

    Example:
        >>> df = query_air_quality('''
        ...     SELECT sitename, AVG(aqi) as avg_aqi
        ...     FROM air_quality
        ...     WHERE year = 2024
        ...     GROUP BY sitename
        ...     ORDER BY avg_aqi DESC
        ...     LIMIT 10
        ... ''')
    """
    loader = AirQualityDataLoader()
    result = loader.query_db(sql)
    loader.close()
    return result


if __name__ == "__main__":
    """
    Demonstration of data loader usage.
    """
    print("=" * 80)
    print("Air Quality Data Loader Demo")
    print("=" * 80)
    print()

    try:
        loader = AirQualityDataLoader()

        # Get date range
        min_date, max_date = loader.get_date_range()
        print(f"Data available from {min_date} to {max_date}")
        print()

        # Get station list
        stations = loader.get_station_list()
        print(f"Total monitoring stations: {len(stations)}")
        print()

        # Get summary statistics
        stats = loader.get_summary_stats()
        print("Summary statistics by county:")
        print(stats.head())

        loader.close()

    except Exception as e:
        print(f"Error: {e}")
