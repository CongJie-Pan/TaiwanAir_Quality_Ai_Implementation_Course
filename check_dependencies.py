#!/usr/bin/env python3
"""
Dependency Checker

Checks if all required dependencies are installed.

Author: Claude Code
Date: 2025-10-13
"""

import sys

def check_dependencies():
    """
    Check if all required dependencies are installed.

    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    required_packages = {
        'pandas': 'pandas',
        'pyarrow': 'pyarrow',
        'duckdb': 'duckdb',
        'numpy': 'numpy'
    }

    missing = []
    installed = []

    print("Checking dependencies...")
    print("=" * 60)

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"✓ {package_name:20s} - installed")
        except ImportError:
            missing.append(package_name)
            print(f"✗ {package_name:20s} - MISSING")

    print("=" * 60)

    if missing:
        print(f"\n❌ Missing {len(missing)} package(s):")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nTo install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All {len(installed)} required packages are installed!")
        return True


if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)
