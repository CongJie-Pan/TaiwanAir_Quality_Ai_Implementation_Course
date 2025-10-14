"""
Unit tests to verify UI scroll safeguards and deprecation cleanup.

These tests are file-content assertions and do not require external packages.
They help prevent regressions that could lock the main view from scrolling or
reintroduce deprecated Streamlit keyword arguments.
"""

import unittest
from pathlib import Path


class TestAppCssAndDeprecatedArgs(unittest.TestCase):
    def test_scroll_css_present_in_app(self):
        app_path = Path('src/main/python/app.py')
        text = app_path.read_text(encoding='utf-8')

        self.assertIn('[data-testid="stAppViewContainer"]', text)
        # Accept either overflow: auto or overflow-y: auto patterns
        self.assertTrue(
            ('overflow: auto' in text) or ('overflow-y: auto' in text),
            'Expected overflow auto rule not found in app CSS'
        )
        self.assertIn('html, body, #root, .stApp', text)

    def test_no_use_container_width_in_pages(self):
        pages_dir = Path('src/main/python/pages')
        for py in pages_dir.glob('*.py'):
            content = py.read_text(encoding='utf-8')
            self.assertNotIn('use_container_width', content, f"Deprecated arg found in {py}")

    def test_nav_toggle_present(self):
        app_path = Path('src/main/python/app.py')
        text = app_path.read_text(encoding='utf-8')
        self.assertIn('USE_TOP_NAV', text)
        self.assertIn('st.sidebar.radio', text)
        self.assertIn('st_navbar', text)


if __name__ == '__main__':
    unittest.main()
