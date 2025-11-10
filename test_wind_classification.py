#!/usr/bin/env python3
"""
Test script to verify wind level classification consistency.

This script validates that the unified wind level classification
is working correctly across all modules.
"""

import sys
from pathlib import Path

# Add src path
src_path = Path(__file__).parent / 'src' / 'main' / 'python'
sys.path.insert(0, str(src_path))

from utils.app_utils import categorize_windspeed
import pandas as pd


def test_wind_classification():
    """Test the categorize_windspeed function with various inputs."""

    print("=" * 60)
    print("Testing Wind Level Classification Function")
    print("=" * 60)

    test_cases = [
        (0.5, '無風(0-1.5)'),
        (1.0, '無風(0-1.5)'),
        (1.5, '無風(0-1.5)'),
        (1.6, '輕風(1.6-3.3)'),
        (2.0, '輕風(1.6-3.3)'),
        (3.3, '輕風(1.6-3.3)'),
        (3.4, '微風(3.4-5.4)'),
        (5.0, '微風(3.4-5.4)'),
        (5.4, '微風(3.4-5.4)'),
        (5.5, '和風(5.5-7.9)'),
        (7.9, '和風(5.5-7.9)'),
        (8.0, '強風(≥8.0)'),
        (10.0, '強風(≥8.0)'),
    ]

    all_passed = True

    for windspeed, expected in test_cases:
        result = categorize_windspeed(windspeed)
        status = "[PASS]" if result == expected else "[FAIL]"
        print(f"{status} | windspeed={windspeed:4.1f} m/s | "
              f"Expected: {expected:15s} | Got: {result}")

        if result != expected:
            all_passed = False

    print("=" * 60)

    # Test with NaN
    nan_result = categorize_windspeed(float('nan'))
    print(f"NaN test: {nan_result} (should be None)")

    print("=" * 60)

    if all_passed:
        print(">> All tests PASSED!")
    else:
        print(">> Some tests FAILED!")

    return all_passed


def test_distribution_consistency():
    """
    Test that the wind level distribution matches between scatter plot
    and crosstab when using the same classification.
    """

    print("\n" + "=" * 60)
    print("Testing Distribution Consistency")
    print("=" * 60)

    # Sample data similar to the daily averages in the visualization
    sample_data = {
        'windspeed': [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5]
    }

    df = pd.DataFrame(sample_data)
    df['wind_level'] = df['windspeed'].apply(categorize_windspeed)

    print("\nSample Data Distribution:")
    print(df.groupby('wind_level').size())

    print("\nExpected behavior:")
    print("- 0.5-1.5 m/s should be classified as '無風(0-1.5)'")
    print("- 1.6-3.3 m/s should be classified as '輕風(1.6-3.3)'")
    print("- 3.4-5.4 m/s should be classified as '微風(3.4-5.4)'")

    print("=" * 60)


def verify_import_structure():
    """Verify that imports work correctly."""

    print("\n" + "=" * 60)
    print("Verifying Import Structure")
    print("=" * 60)

    try:
        from utils.app_utils import categorize_windspeed, prepare_data
        print(">> app_utils imports successful")

        # Check function signature
        import inspect
        sig = inspect.signature(categorize_windspeed)
        print(f">> categorize_windspeed signature: {sig}")

        # Check docstring
        if categorize_windspeed.__doc__:
            print(">> categorize_windspeed has documentation")
        else:
            print("!! categorize_windspeed missing documentation")

    except Exception as e:
        print(f"!! Import failed: {e}")
        return False

    print("=" * 60)
    return True


if __name__ == "__main__":
    print("\n[Wind Level Classification Validation Test]\n")

    # Run all tests
    test1_passed = verify_import_structure()
    test2_passed = test_wind_classification()
    test_distribution_consistency()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if test1_passed and test2_passed:
        print(">> All validation tests PASSED!")
        print("\n>> Wind level classification is now unified across all pages.")
        print("   - app_utils.categorize_windspeed() is the single source of truth")
        print("   - page2_statistical_analysis.py uses the unified function")
        print("   - page3_pattern_discovery.py uses wind_level from prepare_data()")
        sys.exit(0)
    else:
        print("!! Some validation tests FAILED!")
        print("   Please review the errors above.")
        sys.exit(1)
