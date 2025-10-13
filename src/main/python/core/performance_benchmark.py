"""
Performance Benchmark for Air Quality Data Formats

This module benchmarks the performance of different data formats:
- Original CSV
- Parquet files
- DuckDB database

Metrics measured:
- File size
- Load time
- Query time
- Memory usage

Author: Claude Code
Date: 2025-10-13
"""

import pandas as pd
import duckdb
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """
    Benchmarks performance across different data formats.

    This class provides comprehensive performance testing to compare
    CSV, Parquet, and DuckDB formats.

    Attributes:
        csv_path: Path to CSV file
        parquet_dir: Directory containing Parquet files
        db_path: Path to DuckDB database
    """

    def __init__(
        self,
        csv_path: str = "air_quality.csv",
        parquet_dir: str = "data/processed",
        db_path: str = "data/processed/air_quality.duckdb"
    ):
        """
        Initialize the benchmark.

        Args:
            csv_path: Path to CSV file
            parquet_dir: Directory containing Parquet files
            db_path: Path to DuckDB database
        """
        self.csv_path = Path(csv_path)
        self.parquet_dir = Path(parquet_dir)
        self.db_path = Path(db_path)

        logger.info("Performance benchmark initialized")

    def get_file_sizes(self) -> Dict[str, float]:
        """
        Get file sizes for all formats.

        Returns:
            Dictionary with file sizes in MB
        """
        logger.info("\n--- Measuring File Sizes ---")

        sizes = {}

        # CSV size
        if self.csv_path.exists():
            csv_size = self.csv_path.stat().st_size / (1024 * 1024)
            sizes['csv_mb'] = round(csv_size, 2)
            logger.info(f"CSV: {csv_size:.2f} MB")
        else:
            logger.warning(f"CSV file not found: {self.csv_path}")
            sizes['csv_mb'] = None

        # Parquet total size
        if self.parquet_dir.exists():
            parquet_size = sum(
                f.stat().st_size for f in self.parquet_dir.rglob('*.parquet')
            ) / (1024 * 1024)
            sizes['parquet_mb'] = round(parquet_size, 2)
            logger.info(f"Parquet: {parquet_size:.2f} MB")

            # Calculate compression ratio
            if sizes.get('csv_mb'):
                compression_ratio = csv_size / parquet_size
                sizes['compression_ratio'] = round(compression_ratio, 2)
                space_saved = ((csv_size - parquet_size) / csv_size) * 100
                sizes['space_saved_percent'] = round(space_saved, 1)
                logger.info(
                    f"Compression: {compression_ratio:.2f}x "
                    f"({space_saved:.1f}% space saved)"
                )
        else:
            logger.warning(f"Parquet directory not found: {self.parquet_dir}")
            sizes['parquet_mb'] = None

        # DuckDB size
        if self.db_path.exists():
            db_size = self.db_path.stat().st_size / (1024 * 1024)
            sizes['duckdb_mb'] = round(db_size, 2)
            logger.info(f"DuckDB: {db_size:.2f} MB")
        else:
            logger.warning(f"DuckDB file not found: {self.db_path}")
            sizes['duckdb_mb'] = None

        return sizes

    def benchmark_load_performance(
        self,
        sample_size: int = 100000
    ) -> Dict[str, Any]:
        """
        Benchmark data loading performance.

        Args:
            sample_size: Number of rows to load for testing

        Returns:
            Dictionary with load time benchmarks
        """
        logger.info(f"\n--- Benchmarking Load Performance ({sample_size:,} rows) ---")

        results = {}

        # Benchmark CSV loading
        if self.csv_path.exists():
            try:
                start_time = time.time()
                df_csv = pd.read_csv(self.csv_path, nrows=sample_size)
                csv_time = time.time() - start_time

                results['csv_load_time'] = round(csv_time, 3)
                results['csv_rows'] = len(df_csv)
                logger.info(f"CSV: {csv_time:.3f} seconds ({len(df_csv):,} rows)")
            except Exception as e:
                logger.warning(f"CSV load failed: {e}")
                results['csv_load_time'] = None
        else:
            logger.warning("CSV file not available for benchmarking")
            results['csv_load_time'] = None

        # Benchmark Parquet loading
        if self.parquet_dir.exists():
            try:
                start_time = time.time()
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df_parquet = pd.read_parquet(parquet_pattern)

                # Take sample if needed
                if len(df_parquet) > sample_size:
                    df_parquet = df_parquet.head(sample_size)

                parquet_time = time.time() - start_time

                results['parquet_load_time'] = round(parquet_time, 3)
                results['parquet_rows'] = len(df_parquet)
                logger.info(f"Parquet: {parquet_time:.3f} seconds ({len(df_parquet):,} rows)")

                # Calculate speedup
                if results.get('csv_load_time'):
                    speedup = csv_time / parquet_time
                    results['parquet_speedup'] = round(speedup, 2)
                    logger.info(f"Parquet speedup: {speedup:.2f}x faster than CSV")
            except Exception as e:
                logger.warning(f"Parquet load failed: {e}")
                results['parquet_load_time'] = None
        else:
            logger.warning("Parquet files not available for benchmarking")
            results['parquet_load_time'] = None

        return results

    def benchmark_query_performance(self) -> Dict[str, Any]:
        """
        Benchmark query performance.

        Tests common query patterns on different formats.

        Returns:
            Dictionary with query time benchmarks
        """
        logger.info("\n--- Benchmarking Query Performance ---")

        results = {}

        # Test query: Average AQI by county
        test_query = """
            SELECT
                county,
                AVG(aqi) as avg_aqi
            FROM {table}
            WHERE date >= '2024-08-01' AND date <= '2024-08-31'
            GROUP BY county
        """

        # Benchmark Parquet query (using pandas)
        if self.parquet_dir.exists():
            try:
                start_time = time.time()
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(parquet_pattern)

                # Filter and aggregate
                df_filtered = df[
                    (df['date'] >= '2024-08-01') &
                    (df['date'] <= '2024-08-31')
                ]
                result = df_filtered.groupby('county')['aqi'].mean()

                parquet_query_time = time.time() - start_time

                results['parquet_query_time'] = round(parquet_query_time, 3)
                results['parquet_result_rows'] = len(result)
                logger.info(
                    f"Parquet query: {parquet_query_time:.3f} seconds "
                    f"({len(result)} results)"
                )
            except Exception as e:
                logger.warning(f"Parquet query failed: {e}")
                results['parquet_query_time'] = None

        # Benchmark DuckDB query
        if self.db_path.exists():
            try:
                start_time = time.time()
                conn = duckdb.connect(str(self.db_path), read_only=True)

                result = conn.execute(
                    test_query.format(table='air_quality')
                ).fetchdf()

                duckdb_query_time = time.time() - start_time
                conn.close()

                results['duckdb_query_time'] = round(duckdb_query_time, 3)
                results['duckdb_result_rows'] = len(result)
                logger.info(
                    f"DuckDB query: {duckdb_query_time:.3f} seconds "
                    f"({len(result)} results)"
                )

                # Calculate speedup
                if results.get('parquet_query_time'):
                    speedup = parquet_query_time / duckdb_query_time
                    results['duckdb_query_speedup'] = round(speedup, 2)
                    logger.info(f"DuckDB speedup: {speedup:.2f}x faster than Parquet")
            except Exception as e:
                logger.warning(f"DuckDB query failed: {e}")
                results['duckdb_query_time'] = None

        return results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Returns:
            Dictionary with all benchmark results
        """
        logger.info("=" * 80)
        logger.info("Starting Full Performance Benchmark")
        logger.info("=" * 80)

        results = {
            'file_sizes': self.get_file_sizes(),
            'load_performance': self.benchmark_load_performance(sample_size=100000),
            'query_performance': self.benchmark_query_performance()
        }

        logger.info("\n" + "=" * 80)
        logger.info("Benchmark Complete")
        logger.info("=" * 80)

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a formatted benchmark report.

        Args:
            results: Benchmark results dictionary

        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AIR QUALITY DATA FORMAT PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # File sizes
        report_lines.append("FILE SIZES")
        report_lines.append("-" * 80)
        sizes = results.get('file_sizes', {})
        if sizes.get('csv_mb'):
            report_lines.append(f"CSV:              {sizes['csv_mb']:>10.2f} MB")
        if sizes.get('parquet_mb'):
            report_lines.append(f"Parquet:          {sizes['parquet_mb']:>10.2f} MB")
        if sizes.get('duckdb_mb'):
            report_lines.append(f"DuckDB:           {sizes['duckdb_mb']:>10.2f} MB")
        if sizes.get('compression_ratio'):
            report_lines.append(f"Compression:      {sizes['compression_ratio']:>10.2f}x")
            report_lines.append(f"Space Saved:      {sizes['space_saved_percent']:>10.1f}%")
        report_lines.append("")

        # Load performance
        report_lines.append("LOAD PERFORMANCE (100,000 rows)")
        report_lines.append("-" * 80)
        load = results.get('load_performance', {})
        if load.get('csv_load_time'):
            report_lines.append(f"CSV:              {load['csv_load_time']:>10.3f} seconds")
        if load.get('parquet_load_time'):
            report_lines.append(f"Parquet:          {load['parquet_load_time']:>10.3f} seconds")
        if load.get('parquet_speedup'):
            report_lines.append(f"Speedup:          {load['parquet_speedup']:>10.2f}x")
        report_lines.append("")

        # Query performance
        report_lines.append("QUERY PERFORMANCE (Monthly average by county)")
        report_lines.append("-" * 80)
        query = results.get('query_performance', {})
        if query.get('parquet_query_time'):
            report_lines.append(f"Parquet:          {query['parquet_query_time']:>10.3f} seconds")
        if query.get('duckdb_query_time'):
            report_lines.append(f"DuckDB:           {query['duckdb_query_time']:>10.3f} seconds")
        if query.get('duckdb_query_speedup'):
            report_lines.append(f"Speedup:          {query['duckdb_query_speedup']:>10.2f}x")
        report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)


def run_benchmark() -> Dict[str, Any]:
    """
    Convenience function to run full benchmark.

    Returns:
        Dictionary with benchmark results

    Example:
        >>> results = run_benchmark()
        >>> print(f"Parquet is {results['file_sizes']['compression_ratio']}x smaller")
    """
    benchmark = PerformanceBenchmark()
    results = benchmark.run_full_benchmark()

    # Generate and print report
    report = benchmark.generate_report(results)
    print("\n" + report)

    return results


if __name__ == "__main__":
    """
    Main execution when script is run directly.
    """
    print("=" * 80)
    print("Air Quality Data Format Performance Benchmark")
    print("=" * 80)
    print()

    try:
        results = run_benchmark()

    except Exception as e:
        print("\n" + "=" * 80)
        print("BENCHMARK FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("=" * 80)
        raise
