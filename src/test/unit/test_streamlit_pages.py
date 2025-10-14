"""
Unit tests for Streamlit pages and app utilities to prevent UI freezes
and ensure deprecated arguments are not used.

These tests verify that:
- Time period labels are unique (no pandas categorical duplicate-label error)
- Page rendering passes Arrow-compatible DataFrames to st.dataframe
- Pages do not use deprecated `use_container_width` (use `width='stretch'` instead)

Author: Claude Code
Date: 2025-10-14
"""

import unittest
from pathlib import Path
import sys
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    raise unittest.SkipTest(f"Skipping test_streamlit_pages due to missing dependencies: {e}")

# Add project root src path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'main' / 'python'))

from utils import app_utils
from pages import page1_data_overview as p1
from pages import page2_statistical_analysis as p2


class FakeContext:
    """Simple context manager for with-blocks used by Streamlit layout helpers."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSt:
    """
    Lightweight stub of the streamlit API used in pages to capture calls and
    assert that no deprecated args are used. It also avoids real rendering
    to prevent UI-related hangs during tests.
    """

    def __init__(self):
        self.dataframe_calls = []
        self.plotly_chart_calls = []

    # Layout primitives
    def header(self, *args, **kwargs):
        pass

    def markdown(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def text(self, *args, **kwargs):
        pass

    def metric(self, *args, **kwargs):
        pass

    # Containers
    def columns(self, spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [FakeContext() for _ in range(n)]

    def tabs(self, names):
        return [FakeContext() for _ in names]

    def expander(self, *args, **kwargs):
        return FakeContext()

    # Widgets
    def select_slider(self, label, options, value, **kwargs):
        return value

    def selectbox(self, label, options, **kwargs):
        return options[0] if options else None

    def slider(self, label, *args, **kwargs):
        # Support both (min, max, value) positional and keyword value
        if 'value' in kwargs:
            return kwargs['value']
        if len(args) >= 3:
            return args[2]
        if len(args) == 2:
            # return midpoint if min/max provided
            return (args[0] + args[1]) // 2
        return None

    def download_button(self, *args, **kwargs):
        pass

    # Charts/tables
    def dataframe(self, df, **kwargs):
        self.dataframe_calls.append({'df': df, 'kwargs': kwargs})

    def plotly_chart(self, fig, **kwargs):
        self.plotly_chart_calls.append({'fig': fig, 'kwargs': kwargs})

    def bar_chart(self, *args, **kwargs):
        pass


def make_sample_df(rows: int = 48) -> pd.DataFrame:
    """Create a small, valid DataFrame similar to the app input."""
    dates = pd.date_range('2024-08-01', periods=rows, freq='H')
    df = pd.DataFrame({
        'date': dates,
        'sitename': np.random.choice(['S1', 'S2'], size=rows),
        'county': np.random.choice(['台北市', '新北市'], size=rows),
        'aqi': np.random.uniform(20, 180, size=rows).round(1),
        'pollutant': np.random.choice(['PM2.5', 'PM10', 'O3', 'NO2'], size=rows),
        'status': np.random.choice(['Good', 'Moderate'], size=rows),
        'pm2.5': np.random.uniform(5, 80, size=rows).round(1),
        'pm10': np.random.uniform(10, 120, size=rows).round(1),
        'o3': np.random.uniform(5, 100, size=rows).round(1),
        'co': np.random.uniform(0.1, 1.5, size=rows).round(2),
        'so2': np.random.uniform(0.1, 10.0, size=rows).round(2),
        'no2': np.random.uniform(2, 60, size=rows).round(1),
        'windspeed': np.random.uniform(0, 10, size=rows).round(1),
        'winddirec': np.random.uniform(0, 360, size=rows).round(1),
        'longitude': 121 + np.random.uniform(-0.5, 0.5, size=rows),
        'latitude': 25 + np.random.uniform(-0.5, 0.5, size=rows),
    })
    return df


class TestAppUtilsAndPages(unittest.TestCase):
    """Unit tests for app utilities and Streamlit pages rendering."""

    def test_prepare_data_time_period_labels_unique(self):
        """
        Ensure time_period uses unique labels to avoid pandas categorical errors.
        """
        df = make_sample_df(24)
        result = app_utils.prepare_data(df)
        self.assertIn('time_period', result.columns)
        cats = result['time_period'].cat.categories
        # Categories must be unique and expected 6 periods
        self.assertEqual(len(cats), len(set(cats)))
        self.assertGreaterEqual(len(cats), 4)

    def test_page1_data_overview_arrow_compatible_and_no_deprecated_width(self):
        """
        Rendering page1 should pass Arrow-compatible dtypes in the type table
        and not use deprecated `use_container_width` in Streamlit calls.
        """
        df = app_utils.prepare_data(make_sample_df(48))

        fake_st = FakeSt()
        # Monkeypatch the module's streamlit reference
        original_st = p1.st
        try:
            p1.st = fake_st
            p1.render(df)
        finally:
            p1.st = original_st

        # Assert no deprecated arg usage
        for call in fake_st.dataframe_calls + fake_st.plotly_chart_calls:
            self.assertNotIn('use_container_width', call['kwargs'])

        # Find the dtype info table and ensure dtype values are strings
        dtype_tables = [c['df'] for c in fake_st.dataframe_calls if '數據類型' in c['df'].columns]
        self.assertTrue(len(dtype_tables) >= 1)
        dtype_col = dtype_tables[0]['數據類型']
        self.assertTrue(all(isinstance(v, str) for v in dtype_col.tolist()))

    def test_page2_no_deprecated_width(self):
        """
        Rendering page2 should not use deprecated `use_container_width`.
        """
        df = app_utils.prepare_data(make_sample_df(72))

        fake_st = FakeSt()
        original_st = p2.st
        try:
            p2.st = fake_st
            p2.render(df)
        finally:
            p2.st = original_st

        for call in fake_st.dataframe_calls + fake_st.plotly_chart_calls:
            self.assertNotIn('use_container_width', call['kwargs'])


if __name__ == '__main__':
    unittest.main()
