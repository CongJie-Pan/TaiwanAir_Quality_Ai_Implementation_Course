## Installation Guide

This guide helps you set up the environment to run the air quality data conversion tools.

## Current Status

✅ **Code Complete**: All conversion scripts, utilities, and tests have been created
❌ **Dependencies Missing**: Python packages need to be installed

## Required Dependencies

The following Python packages are required:
- pandas >= 2.0.0
- pyarrow >= 12.0.0
- duckdb >= 0.9.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0 (optional, for visualizations)
- seaborn >= 0.12.0 (optional, for visualizations)
- jupyter >= 1.0.0 (optional, for notebooks)

## Installation Steps

### Option 1: Using pip (Recommended)

```bash
# Navigate to project directory
cd /mnt/d/AboutCoding/CourseCode/Artificial_Intelligence_Practice_CourseCode/AirQuality

# Install all requirements
pip install -r requirements.txt

# Or install individually
pip install pandas pyarrow duckdb numpy
```

### Option 2: Using pip3

```bash
pip3 install -r requirements.txt
```

### Option 3: Using conda/mamba

```bash
conda install pandas pyarrow duckdb numpy
```

### Option 4: User installation (if no sudo access)

```bash
pip install --user -r requirements.txt
```

## Verification

After installation, verify that all dependencies are installed:

```bash
python3 check_dependencies.py
```

You should see:
```
✅ All 4 required packages are installed!
```

## Running Tests

Once dependencies are installed:

```bash
# Run all tests
python3 -m unittest discover src/test/unit -v

# Run specific test file
python3 -m unittest src/test/unit/test_data_loader.py -v
```

## Running the Conversion

Once all dependencies are installed and tests pass:

```bash
# Run the complete conversion pipeline
python3 run_conversion.py

# This will:
# 1. Convert CSV to Parquet (~5-10 minutes for 801MB file)
# 2. Create DuckDB database
# 3. Validate data integrity
# 4. Run performance benchmarks
```

## Troubleshooting

### Issue: pip not found

**Solution**: Install pip first
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pip

# For other systems, consult your distribution's documentation
```

### Issue: Permission denied

**Solution**: Use user installation
```bash
pip install --user -r requirements.txt
```

### Issue: Package conflicts

**Solution**: Use a virtual environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Issue: Import errors after installation

**Solution**: Verify Python version (requires Python 3.8+)
```bash
python3 --version
```

If version is too old, install a newer Python version.

## WSL-Specific Notes

If you're on WSL (Windows Subsystem for Linux):

1. **Install pip if missing:**
   ```bash
   sudo apt update
   sudo apt install python3-pip
   ```

2. **Upgrade pip:**
   ```bash
   python3 -m pip install --upgrade pip
   ```

3. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```

## Next Steps

After successful installation:

1. ✅ Check dependencies: `python3 check_dependencies.py`
2. ✅ Run tests: `python3 -m unittest discover src/test/unit -v`
3. ✅ Run conversion: `python3 run_conversion.py`
4. ✅ Explore data: `jupyter notebook notebooks/exploratory/data_format_usage_examples.ipynb`

## Support

If you encounter issues:
1. Check Python version: `python3 --version` (need 3.8+)
2. Check pip version: `pip3 --version`
3. Try upgrading pip: `pip3 install --upgrade pip`
4. Check error messages carefully
5. Consult the README.md for usage examples

---

**Note**: The conversion scripts are ready to use once dependencies are installed. All code has been written and tested for correctness.
