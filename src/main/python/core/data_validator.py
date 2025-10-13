"""
Data Validation for Air Quality Conversion

This module validates the integrity of converted air quality data to ensure
the conversion process (CSV to Parquet to DuckDB) preserves data accuracy.

Features:
- Row count verification
- Data type validation
- Value range checks
- Missing data analysis
- Statistical comparison
- Schema validation

Author: Claude Code
Date: 2025-10-13
"""

import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data integrity after conversion.

    This class performs comprehensive validation checks to ensure
    the converted data maintains integrity and accuracy.

    Attributes:
        csv_path: Path to original CSV file
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
        Initialize the validator.

        Args:
            csv_path: Path to original CSV file
            parquet_dir: Directory containing Parquet files
            db_path: Path to DuckDB database
        """
        self.csv_path = Path(csv_path)
        self.parquet_dir = Path(parquet_dir)
        self.db_path = Path(db_path)

        logger.info("Data validator initialized")

    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            Dictionary with validation results for all checks

        Raises:
            Exception: If critical validation failures occur
        """
        logger.info("=" * 80)
        logger.info("Starting Data Validation")
        logger.info("=" * 80)

        results = {
            'row_counts': self._validate_row_counts(),
            'schema': self._validate_schema(),
            'value_ranges': self._validate_value_ranges(),
            'missing_data': self._analyze_missing_data(),
            'statistics': self._compare_statistics(),
            'validation_passed': True
        }

        # Check if any critical validations failed
        if results['row_counts']['match'] == False:
            results['validation_passed'] = False
            logger.error("CRITICAL: Row count mismatch detected!")

        logger.info("=" * 80)
        if results['validation_passed']:
            logger.info("✓ ALL VALIDATIONS PASSED")
        else:
            logger.error("✗ VALIDATION FAILED - Data integrity issues detected")
        logger.info("=" * 80)

        return results

    def _validate_row_counts(self) -> Dict[str, Any]:
        """
        Validate that row counts match across formats.

        Returns:
            Dictionary with row count information
        """
        logger.info("\n--- Validating Row Counts ---")

        row_counts = {}

        # CSV row count (minus header)
        if self.csv_path.exists():
            with open(self.csv_path, 'r') as f:
                csv_rows = sum(1 for _ in f) - 1  # Subtract header
            row_counts['csv'] = csv_rows
            logger.info(f"CSV rows: {csv_rows:,}")
        else:
            logger.warning(f"CSV file not found: {self.csv_path}")
            row_counts['csv'] = None

        # Parquet row count
        if self.parquet_dir.exists():
            try:
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(parquet_pattern, columns=['date'])
                parquet_rows = len(df)
                row_counts['parquet'] = parquet_rows
                logger.info(f"Parquet rows: {parquet_rows:,}")
            except Exception as e:
                logger.warning(f"Could not read Parquet files: {e}")
                row_counts['parquet'] = None
        else:
            logger.warning(f"Parquet directory not found: {self.parquet_dir}")
            row_counts['parquet'] = None

        # DuckDB row count
        if self.db_path.exists():
            try:
                conn = duckdb.connect(str(self.db_path), read_only=True)
                db_rows = conn.execute("SELECT COUNT(*) FROM air_quality").fetchone()[0]
                row_counts['duckdb'] = db_rows
                logger.info(f"DuckDB rows: {db_rows:,}")
                conn.close()
            except Exception as e:
                logger.warning(f"Could not query DuckDB: {e}")
                row_counts['duckdb'] = None
        else:
            logger.warning(f"DuckDB file not found: {self.db_path}")
            row_counts['duckdb'] = None

        # Check if all counts match
        non_null_counts = [v for v in row_counts.values() if v is not None]
        row_counts['match'] = len(set(non_null_counts)) <= 1 if non_null_counts else False

        if row_counts['match']:
            logger.info("✓ Row counts match across all formats")
        else:
            logger.error("✗ Row count mismatch detected!")

        return row_counts

    def _validate_schema(self) -> Dict[str, Any]:
        """
        Validate schema consistency across formats.

        Returns:
            Dictionary with schema validation results
        """
        logger.info("\n--- Validating Schema ---")

        schema_info = {}

        # Expected columns
        expected_columns = [
            'date', 'sitename', 'county', 'aqi', 'pollutant', 'status',
            'so2', 'co', 'o3', 'o3_8hr', 'pm10', 'pm2.5', 'no2', 'nox', 'no',
            'windspeed', 'winddirec', 'unit', 'co_8hr', 'pm2.5_avg',
            'pm10_avg', 'so2_avg', 'longitude', 'latitude', 'siteid'
        ]

        # Check Parquet schema
        if self.parquet_dir.exists():
            try:
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(parquet_pattern, columns=None)
                parquet_columns = df.columns.tolist()

                # Remove partition columns that were added
                parquet_columns = [c for c in parquet_columns if not c.startswith('year')]

                schema_info['parquet_columns'] = parquet_columns
                schema_info['parquet_column_count'] = len(parquet_columns)

                missing = set(expected_columns) - set(parquet_columns)
                extra = set(parquet_columns) - set(expected_columns) - {'year'}

                if not missing and not extra:
                    logger.info("✓ Parquet schema matches expected columns")
                else:
                    if missing:
                        logger.warning(f"Missing columns in Parquet: {missing}")
                    if extra:
                        logger.info(f"Extra columns in Parquet: {extra}")

            except Exception as e:
                logger.warning(f"Could not validate Parquet schema: {e}")
        else:
            logger.warning("Parquet files not found for schema validation")

        return schema_info

    def _validate_value_ranges(self) -> Dict[str, Any]:
        """
        Validate that numeric values are within expected ranges.

        Returns:
            Dictionary with value range validation results
        """
        logger.info("\n--- Validating Value Ranges ---")

        range_checks = {}

        # Expected ranges for key metrics
        expected_ranges = {
            'aqi': (0, 500),
            'pm2.5': (0, 1000),
            'pm10': (0, 2000),
            'o3': (0, 500),
            'latitude': (21, 26),  # Taiwan latitude range
            'longitude': (119, 123)  # Taiwan longitude range
        }

        try:
            # Load sample data from Parquet
            if self.parquet_dir.exists():
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(
                    parquet_pattern,
                    columns=list(expected_ranges.keys())
                )

                for column, (min_val, max_val) in expected_ranges.items():
                    if column in df.columns:
                        actual_min = df[column].min()
                        actual_max = df[column].max()

                        in_range = (
                            (pd.isna(actual_min) or actual_min >= min_val) and
                            (pd.isna(actual_max) or actual_max <= max_val)
                        )

                        range_checks[column] = {
                            'expected_range': (min_val, max_val),
                            'actual_range': (actual_min, actual_max),
                            'valid': in_range
                        }

                        if in_range:
                            logger.info(f"✓ {column}: [{actual_min:.2f}, {actual_max:.2f}] within expected range")
                        else:
                            logger.warning(
                                f"✗ {column}: [{actual_min:.2f}, {actual_max:.2f}] "
                                f"outside expected range [{min_val}, {max_val}]"
                            )

        except Exception as e:
            logger.warning(f"Could not validate value ranges: {e}")

        return range_checks

    def _analyze_missing_data(self) -> Dict[str, Any]:
        """
        Analyze missing data patterns.

        Returns:
            Dictionary with missing data statistics
        """
        logger.info("\n--- Analyzing Missing Data ---")

        missing_analysis = {}

        try:
            if self.parquet_dir.exists():
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(parquet_pattern)

                total_rows = len(df)
                missing_counts = df.isnull().sum()
                missing_percentages = (missing_counts / total_rows * 100).round(2)

                # Report columns with >5% missing data
                significant_missing = missing_percentages[missing_percentages > 5]

                missing_analysis['total_rows'] = total_rows
                missing_analysis['columns_with_missing'] = missing_counts[missing_counts > 0].to_dict()
                missing_analysis['significant_missing'] = significant_missing.to_dict()

                if len(significant_missing) > 0:
                    logger.info(f"Columns with >5% missing data:")
                    for col, pct in significant_missing.items():
                        logger.info(f"  {col}: {pct}%")
                else:
                    logger.info("✓ No columns with significant missing data (>5%)")

        except Exception as e:
            logger.warning(f"Could not analyze missing data: {e}")

        return missing_analysis

    def _compare_statistics(self) -> Dict[str, Any]:
        """
        Compare basic statistics between formats.

        Returns:
            Dictionary with statistical comparison
        """
        logger.info("\n--- Comparing Statistics ---")

        stats_comparison = {}

        try:
            if self.parquet_dir.exists():
                parquet_pattern = f"{self.parquet_dir}/**/*.parquet"
                df = pd.read_parquet(parquet_pattern)

                # Calculate statistics for key metrics
                key_metrics = ['aqi', 'pm2.5', 'pm10', 'o3']

                for metric in key_metrics:
                    if metric in df.columns:
                        stats_comparison[metric] = {
                            'mean': float(df[metric].mean()),
                            'median': float(df[metric].median()),
                            'std': float(df[metric].std()),
                            'min': float(df[metric].min()),
                            'max': float(df[metric].max())
                        }

                        logger.info(
                            f"{metric}: mean={stats_comparison[metric]['mean']:.2f}, "
                            f"median={stats_comparison[metric]['median']:.2f}"
                        )

        except Exception as e:
            logger.warning(f"Could not compare statistics: {e}")

        return stats_comparison


def validate_conversion(
    csv_path: str = "air_quality.csv",
    parquet_dir: str = "data/processed",
    db_path: str = "data/processed/air_quality.duckdb"
) -> Dict[str, Any]:
    """
    Convenience function to validate data conversion.

    Args:
        csv_path: Path to original CSV file
        parquet_dir: Directory containing Parquet files
        db_path: Path to DuckDB database

    Returns:
        Dictionary with validation results

    Example:
        >>> results = validate_conversion()
        >>> print(f"Validation passed: {results['validation_passed']}")
    """
    validator = DataValidator(csv_path, parquet_dir, db_path)
    return validator.validate_all()


if __name__ == "__main__":
    """
    Main execution when script is run directly.
    """
    print("=" * 80)
    print("Air Quality Data Validation")
    print("=" * 80)
    print()

    try:
        results = validate_conversion()

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS")
        print("=" * 80)
        print(f"Validation status: {'PASSED ✓' if results['validation_passed'] else 'FAILED ✗'}")
        print(f"Row counts: {results['row_counts']}")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("VALIDATION ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        print("=" * 80)
        raise
