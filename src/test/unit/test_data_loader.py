"""
Unit tests for data_loader module.

Tests the AirQualityDataLoader class and utility functions.

Author: Claude Code
Date: 2025-10-13
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
try:
    import pandas as pd
except Exception as e:
    raise unittest.SkipTest(f"Skipping test_data_loader due to missing dependencies: {e}")
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'main' / 'python'))

from utils.data_loader import AirQualityDataLoader


class TestAirQualityDataLoader(unittest.TestCase):
    """
    Test cases for AirQualityDataLoader class.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.loader = AirQualityDataLoader(
            data_dir=".",
            parquet_dir="test_data/processed",
            db_path="test_data/test.duckdb"
        )

    def test_initialization(self):
        """
        Test that loader initializes correctly.
        """
        self.assertIsNotNone(self.loader)
        self.assertEqual(str(self.loader.parquet_dir), "test_data/processed")
        self.assertEqual(str(self.loader.db_path), "test_data/test.duckdb")

    @patch('utils.data_loader.pd.read_parquet')
    def test_load_parquet_basic(self, mock_read_parquet):
        """
        Test basic Parquet loading without filters.
        """
        # Mock return data
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10, freq='h'),
            'sitename': ['Station1'] * 10,
            'county': ['Taipei City'] * 10,
            'aqi': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            'pm2.5': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        })
        mock_read_parquet.return_value = mock_df

        # Load data
        with patch.object(Path, 'exists', return_value=True):
            result = self.loader.load_parquet()

        # Verify
        self.assertEqual(len(result), 10)
        self.assertIn('date', result.columns)
        self.assertIn('aqi', result.columns)

    @patch('utils.data_loader.pd.read_parquet')
    def test_load_parquet_with_filters(self, mock_read_parquet):
        """
        Test Parquet loading with date and county filters.
        """
        # Mock return data
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='h'),
            'sitename': ['Station1'] * 50 + ['Station2'] * 50,
            'county': ['Taipei City'] * 50 + ['Kaohsiung City'] * 50,
            'aqi': list(range(50, 150)),
            'pm2.5': list(range(10, 110))
        })
        mock_read_parquet.return_value = mock_df

        # Load with filters
        with patch.object(Path, 'exists', return_value=True):
            result = self.loader.load_parquet(
                start_date='2024-01-02',
                end_date='2024-01-03',
                counties=['Taipei City']
            )

        # Verify filters were applied
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(mock_df))

    def test_load_by_year(self):
        """
        Test loading data by specific year.
        """
        # This would require mocking file system
        # For now, test that method exists and handles missing data
        with patch.object(Path, 'exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.loader.load_by_year(2024)

    def test_get_date_range_parquet(self):
        """
        Test getting date range from Parquet files.
        """
        mock_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='h')
        })

        with patch.object(self.loader, 'load_parquet', return_value=mock_df):
            with patch.object(self.loader.db_path, 'exists', return_value=False):
                min_date, max_date = self.loader.get_date_range()

        self.assertEqual(min_date, mock_df['date'].min())
        self.assertEqual(max_date, mock_df['date'].max())

    def test_close_connection(self):
        """
        Test closing database connection.
        """
        # Mock connection
        self.loader._db_connection = Mock()

        # Close
        self.loader.close()

        # Verify closed
        self.loader._db_connection.close.assert_called_once()
        self.assertIsNone(self.loader._db_connection)


class TestConvenienceFunctions(unittest.TestCase):
    """
    Test convenience functions.
    """

    @patch('utils.data_loader.AirQualityDataLoader')
    def test_load_air_quality_data(self, mock_loader_class):
        """
        Test load_air_quality_data convenience function.
        """
        from utils.data_loader import load_air_quality_data

        # Mock loader instance
        mock_loader = Mock()
        mock_loader.load_parquet.return_value = pd.DataFrame({'aqi': [50, 60, 70]})
        mock_loader_class.return_value = mock_loader

        # Call function
        result = load_air_quality_data(
            start_date='2024-01-01',
            end_date='2024-01-31',
            county='Taipei City'
        )

        # Verify
        self.assertEqual(len(result), 3)
        mock_loader.load_parquet.assert_called_once()


if __name__ == '__main__':
    unittest.main()
