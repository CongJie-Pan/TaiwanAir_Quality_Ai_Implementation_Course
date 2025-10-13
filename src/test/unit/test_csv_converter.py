"""
Unit tests for CSV to Parquet converter.

Tests the CSVToParquetConverter class.

Author: Claude Code
Date: 2025-10-13
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'main' / 'python'))

from core.csv_to_parquet_converter import CSVToParquetConverter


class TestCSVToParquetConverter(unittest.TestCase):
    """
    Test cases for CSVToParquetConverter class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.test_csv = Path(self.test_dir) / "test.csv"
        self.output_dir = Path(self.test_dir) / "output"

        # Create a small test CSV
        test_data = """date,sitename,county,aqi,pollutant,status,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,unit,co_8hr,pm2.5_avg,pm10_avg,so2_avg,longitude,latitude,siteid
2024-01-01 00:00,TestSite,TestCounty,50.0,PM2.5,Good,1.0,0.2,30.0,35.0,20.0,15.0,5.0,6.0,1.0,2.0,180.0,,0.2,16.0,22.0,1.0,121.0,25.0,1.0
2024-01-01 01:00,TestSite,TestCounty,55.0,PM2.5,Good,1.1,0.21,31.0,36.0,21.0,16.0,5.1,6.1,1.1,2.1,181.0,,0.21,17.0,23.0,1.1,121.0,25.0,1.0"""

        with open(self.test_csv, 'w') as f:
            f.write(test_data)

    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Remove temporary directory
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """
        Test converter initialization.
        """
        converter = CSVToParquetConverter(
            csv_path=str(self.test_csv),
            output_dir=str(self.output_dir)
        )

        self.assertIsNotNone(converter)
        self.assertEqual(converter.csv_path, self.test_csv)
        self.assertEqual(converter.chunk_size, 100000)

    def test_initialization_missing_csv(self):
        """
        Test initialization with missing CSV file.
        """
        with self.assertRaises(FileNotFoundError):
            CSVToParquetConverter(
                csv_path="nonexistent.csv",
                output_dir=str(self.output_dir)
            )

    def test_optimize_datatypes(self):
        """
        Test data type optimization.
        """
        converter = CSVToParquetConverter(
            csv_path=str(self.test_csv),
            output_dir=str(self.output_dir)
        )

        # Create test DataFrame
        df = pd.DataFrame({
            'date': ['2024-01-01 00:00'],
            'sitename': ['TestSite'],
            'county': ['TestCounty'],
            'aqi': [50.0],
            'pm2.5': [15.0]
        })

        # Optimize
        optimized_df = converter._optimize_datatypes(df)

        # Verify date is datetime
        self.assertEqual(optimized_df['date'].dtype, 'datetime64[ns]')

        # Verify categorical columns
        self.assertEqual(optimized_df['sitename'].dtype.name, 'category')
        self.assertEqual(optimized_df['county'].dtype.name, 'category')

    def test_add_partition_columns(self):
        """
        Test adding partition columns.
        """
        converter = CSVToParquetConverter(
            csv_path=str(self.test_csv),
            output_dir=str(self.output_dir)
        )

        # Create test DataFrame with datetime
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='h'),
            'aqi': [50] * 10
        })

        # Add partition columns
        result = converter._add_partition_columns(df)

        # Verify year column was added
        self.assertIn('year', result.columns)
        self.assertEqual(result['year'].iloc[0], 2024)

    def test_get_conversion_info(self):
        """
        Test getting conversion information.
        """
        converter = CSVToParquetConverter(
            csv_path=str(self.test_csv),
            output_dir=str(self.output_dir)
        )

        info = converter.get_conversion_info()

        self.assertIn('csv_path', info)
        self.assertIn('output_dir', info)
        self.assertIn('chunk_size', info)
        self.assertIn('csv_file_size_mb', info)


if __name__ == '__main__':
    unittest.main()
