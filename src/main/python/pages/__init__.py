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

from . import page1_data_overview
from . import page2_statistical_analysis
from . import page3_pattern_discovery
from . import page4_wisdom_decision
from . import page5_prediction_model

__all__ = [
    'page1_data_overview',
    'page2_statistical_analysis',
    'page3_pattern_discovery',
    'page4_wisdom_decision',
    'page5_prediction_model'
]
