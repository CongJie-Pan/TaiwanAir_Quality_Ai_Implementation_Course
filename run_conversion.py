#!/usr/bin/env python3
"""
Master Conversion Script for Air Quality Data

This script orchestrates the complete conversion process:
1. Convert CSV to Parquet format
2. Create DuckDB database
3. Validate data integrity
4. Run performance benchmarks

Usage:
    python3 run_conversion.py [--skip-validation] [--skip-benchmark]

Author: Claude Code
Date: 2025-10-13
"""

import sys
from pathlib import Path
import argparse
import logging

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src' / 'main' / 'python'))

from core.csv_to_parquet_converter import convert_air_quality_csv
from core.duckdb_database_creator import create_air_quality_database
from core.data_validator import validate_conversion
from core.performance_benchmark import run_benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main execution function for data conversion pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Convert air quality CSV data to optimized formats'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    parser.add_argument(
        '--skip-benchmark',
        action='store_true',
        help='Skip performance benchmark'
    )
    parser.add_argument(
        '--csv-path',
        default='air_quality.csv',
        help='Path to CSV file (default: air_quality.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("AIR QUALITY DATA CONVERSION PIPELINE")
    print("=" * 80)
    print()

    try:
        # Step 1: Convert CSV to Parquet
        print("\n" + "=" * 80)
        print("STEP 1: Converting CSV to Parquet")
        print("=" * 80)
        parquet_stats = convert_air_quality_csv(
            csv_path=args.csv_path,
            output_dir=args.output_dir
        )
        print(f"\n[SUCCESS] CSV to Parquet conversion completed")
        print(f"  - Rows: {parquet_stats['total_rows']:,}")
        print(f"  - Size: {parquet_stats['file_size_mb']} MB")
        print(f"  - Time: {parquet_stats['processing_time_seconds']} seconds")

        # Step 2: Create DuckDB database
        print("\n" + "=" * 80)
        print("STEP 2: Creating DuckDB Database")
        print("=" * 80)
        db_stats = create_air_quality_database(
            parquet_dir=args.output_dir,
            db_path=f"{args.output_dir}/air_quality.duckdb"
        )
        print(f"\n[SUCCESS] DuckDB database created")
        print(f"  - Rows: {db_stats['table_count']:,}")
        print(f"  - Views: {', '.join(db_stats['views_created'])}")

        # Step 3: Validate data integrity
        if not args.skip_validation:
            print("\n" + "=" * 80)
            print("STEP 3: Validating Data Integrity")
            print("=" * 80)
            validation_results = validate_conversion(
                csv_path=args.csv_path,
                parquet_dir=args.output_dir,
                db_path=f"{args.output_dir}/air_quality.duckdb"
            )

            if validation_results['validation_passed']:
                print(f"\n[SUCCESS] Data validation passed")
            else:
                print(f"\n[FAILED] Data validation failed - please check logs")
                sys.exit(1)
        else:
            print("\n[SKIP] Skipping validation (--skip-validation)")

        # Step 4: Run performance benchmark
        if not args.skip_benchmark:
            print("\n" + "=" * 80)
            print("STEP 4: Running Performance Benchmark")
            print("=" * 80)
            benchmark_results = run_benchmark()
            print(f"\n[SUCCESS] Performance benchmark completed")
        else:
            print("\n[SKIP] Skipping benchmark (--skip-benchmark)")

        # Summary
        print("\n" + "=" * 80)
        print("CONVERSION PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Explore the data using notebooks/exploratory/data_format_usage_examples.ipynb")
        print("2. Load data in Python: from utils.data_loader import AirQualityDataLoader")
        print("3. Query with SQL via DuckDB database")
        print("\nFor more information, see README.md")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("CONVERSION PIPELINE FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print("\nPlease check the error message above and try again.")
        print("=" * 80)
        logger.exception("Conversion pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
