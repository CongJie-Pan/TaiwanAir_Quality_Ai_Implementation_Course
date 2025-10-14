"""
Pages Module for Air Quality Streamlit Application

This module contains all page implementations for the DIKW hierarchy:
- page1_data_overview: Data Layer - Raw data display
- page2_statistical_analysis: Information Layer - Statistics and trends
- page3_pattern_discovery: Knowledge Layer - Pattern recognition
- page4_wisdom_decision: Wisdom Layer - Decision support
- page5_prediction_model: Wisdom Layer Advanced - Prediction models

Each page implements a render(df) function that displays the page content.

Author: Claude Code
Date: 2025-10-14
"""

"""
Pages Module for Air Quality Streamlit Application

This module contains available page implementations for the DIKW hierarchy.

Note:
- Pages 4 and 5 (Wisdom Decision, Prediction Model) are temporarily hidden
  from the UI and not exported here to avoid accidental imports in runtime.
  To re-enable, import and add them back to __all__.
"""

from . import page1_data_overview
from . import page2_statistical_analysis
from . import page3_pattern_discovery

__all__ = [
    'page1_data_overview',
    'page2_statistical_analysis',
    'page3_pattern_discovery'
]
