"""
CSV to Parquet Converter for Air Quality Data

This module handles the conversion of large CSV air quality datasets to optimized
Parquet format with partitioning by year for efficient storage and query performance.

Features:
- Chunked reading to handle large files without memory overflow
- Automatic data type optimization
- Year-based partitioning for efficient queries
- Progress tracking and logging
- Data validation and error handling

Author: Claude Code
Date: 2025-10-13
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging with English messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CSVToParquetConverter:
    """
    Converts large CSV air quality datasets to partitioned Parquet format.

    This class handles the conversion process with memory-efficient chunked reading,
    automatic data type optimization, and year-based partitioning for better
    query performance.

    Attributes:
        csv_path: Path to the source CSV file
        output_dir: Directory where Parquet files will be saved
        chunk_size: Number of rows to process at once (default: 100,000)
    """

    # Define optimal data types for each column to reduce memory usage
    DTYPE_MAP = {
        'sitename': 'category',
        'county': 'category',
        'aqi': 'float32',
        'pollutant': 'category',
        'status': 'category',
        'so2': 'float32',
        'co': 'float32',
        'o3': 'float32',
        'o3_8hr': 'float32',
        'pm10': 'float32',
        'pm2.5': 'float32',
        'no2': 'float32',
        'nox': 'float32',
        'no': 'float32',
        'windspeed': 'float32',
        'winddirec': 'float32',
        'unit': 'category',
        'co_8hr': 'float32',
        'pm2.5_avg': 'float32',
        'pm10_avg': 'float32',
        'so2_avg': 'float32',
        'longitude': 'float64',
        'latitude': 'float64',
        'siteid': 'float32'
    }

    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        chunk_size: int = 100000
    ):
        """
        Initialize the converter.

        Args:
            csv_path: Path to the source CSV file
            output_dir: Directory where Parquet files will be saved
            chunk_size: Number of rows to process at once (default: 100,000)
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size

        # Validate inputs
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Converter initialized: {self.csv_path} -> {self.output_dir}")

    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame data types to reduce memory usage.

        This method converts columns to more memory-efficient types:
        - String columns with limited unique values -> category
        - Float64 -> Float32 (where precision loss is acceptable)
        - Parse date strings to datetime objects

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with optimized data types
        """
        logger.debug("Optimizing data types...")

        # Convert date column to datetime with mixed format support
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

        # Apply predefined dtype optimizations
        for col, dtype in self.DTYPE_MAP.items():
            if col in df.columns:
                try:
                    if dtype in ['float32', 'float64']:
                        # Use pd.to_numeric for numeric columns to handle errors gracefully
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to {dtype}: {e}")

        return df

    def _add_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add partition columns for efficient data organization.

        Extracts year from the date column to enable partitioning.

        Args:
            df: Input DataFrame with 'date' column

        Returns:
            DataFrame with added 'year' column
        """
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
        else:
            logger.warning("No 'date' column found, skipping partition columns")

        return df

    def convert(self) -> Dict[str, Any]:
        """
        Convert CSV to partitioned Parquet format.

        This method performs the main conversion process:
        1. Reads CSV in chunks to manage memory
        2. Optimizes data types for each chunk
        3. Adds partitioning columns
        4. Writes to Parquet with year-based partitioning
        5. Tracks and reports statistics

        Returns:
            Dictionary containing conversion statistics:
            - total_rows: Total number of rows processed
            - partitions: List of years created as partitions
            - file_size_mb: Total size of output files in MB
            - processing_time_seconds: Time taken for conversion

        Raises:
            Exception: If conversion fails
        """
        logger.info("Starting CSV to Parquet conversion...")
        start_time = datetime.now()

        total_rows = 0
        partitions_created = set()

        try:
            # Initialize Parquet writer (will append chunks)
            parquet_writer = None
            schema = None

            # Read and process CSV in chunks
            chunk_iterator = pd.read_csv(
                self.csv_path,
                chunksize=self.chunk_size,
                dtype={col: str for col in self.DTYPE_MAP.keys() if col != 'date'},
                na_values=['-', '', 'NA', 'N/A', 'null', 'NULL']
            )

            for chunk_num, chunk in enumerate(chunk_iterator, 1):
                # Optimize data types
                chunk = self._optimize_datatypes(chunk)

                # Add partition columns
                chunk = self._add_partition_columns(chunk)

                # Track partitions
                if 'year' in chunk.columns:
                    partitions_created.update(chunk['year'].unique())

                # Convert to PyArrow Table
                table = pa.Table.from_pandas(chunk)

                # Initialize schema from first chunk
                if schema is None:
                    schema = table.schema

                # Write to Parquet with partitioning
                pq.write_to_dataset(
                    table,
                    root_path=str(self.output_dir),
                    partition_cols=['year'] if 'year' in chunk.columns else None,
                    existing_data_behavior='overwrite_or_ignore'
                )

                total_rows += len(chunk)

                # Log progress
                if chunk_num % 10 == 0:
                    logger.info(
                        f"Processed {chunk_num} chunks, "
                        f"{total_rows:,} rows total"
                    )

            # Calculate output file size
            file_size_mb = sum(
                f.stat().st_size for f in self.output_dir.rglob('*.parquet')
            ) / (1024 * 1024)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Prepare statistics
            stats = {
                'total_rows': total_rows,
                'partitions': sorted(list(partitions_created)),
                'file_size_mb': round(file_size_mb, 2),
                'processing_time_seconds': round(processing_time, 2),
                'output_dir': str(self.output_dir)
            }

            logger.info("Conversion completed successfully!")
            logger.info(f"Statistics: {stats}")

            return stats

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise

    def get_conversion_info(self) -> Dict[str, Any]:
        """
        Get information about the conversion configuration.

        Returns:
            Dictionary with converter configuration details
        """
        return {
            'csv_path': str(self.csv_path),
            'output_dir': str(self.output_dir),
            'chunk_size': self.chunk_size,
            'csv_file_size_mb': round(
                self.csv_path.stat().st_size / (1024 * 1024), 2
            )
        }


def convert_air_quality_csv(
    csv_path: str = "air_quality.csv",
    output_dir: str = "data/processed",
    chunk_size: int = 100000
) -> Dict[str, Any]:
    """
    Convenience function to convert air quality CSV to Parquet.

    This is a simplified interface to the CSVToParquetConverter class.

    Args:
        csv_path: Path to the source CSV file (default: "air_quality.csv")
        output_dir: Directory where Parquet files will be saved (default: "data/processed")
        chunk_size: Number of rows to process at once (default: 100,000)

    Returns:
        Dictionary containing conversion statistics

    Example:
        >>> stats = convert_air_quality_csv()
        >>> print(f"Converted {stats['total_rows']:,} rows")
        >>> print(f"Output size: {stats['file_size_mb']} MB")
    """
    converter = CSVToParquetConverter(csv_path, output_dir, chunk_size)

    # Log converter info
    info = converter.get_conversion_info()
    logger.info(f"Converter configuration: {info}")

    # Perform conversion
    stats = converter.convert()

    return stats


if __name__ == "__main__":
    """
    Main execution when script is run directly.

    This will convert the default air_quality.csv file to Parquet format.
    """
    print("=" * 80)
    print("Air Quality CSV to Parquet Converter")
    print("=" * 80)
    print()

    # Run conversion
    try:
        stats = convert_air_quality_csv()

        print("\n" + "=" * 80)
        print("CONVERSION SUCCESSFUL")
        print("=" * 80)
        print(f"Total rows processed: {stats['total_rows']:,}")
        print(f"Year partitions: {', '.join(map(str, stats['partitions']))}")
        print(f"Output size: {stats['file_size_mb']} MB")
        print(f"Processing time: {stats['processing_time_seconds']} seconds")
        print(f"Output directory: {stats['output_dir']}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("CONVERSION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("=" * 80)
        raise
