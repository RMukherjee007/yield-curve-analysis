# Yield Curve Analysis (Indian G-Sec)

A production-grade Python analytics tool to construct, interpolate, and compare Indian government bond (G-Sec) and Treasury Bill yield curves across maturities.

## Overview

The yield curve is a core fixed-income indicator reflecting market expectations about interest rates, inflation, and monetary policy. This package loads Indian G-Sec yields across short-to-long tenors, fits continuous curves, and decomposes curve shifts between dates into parallel level shifts, slope (steepening/flattening), and butterfly curvature changes.

## Key Capabilities

- **Tenor Normalization & Validation**: Parses Indian T-Bill and G-Sec tenors (`91D`, `182D`, `364D`, `3M`, `6M`, `1Y`–`30Y`) into year fractions, sorts maturities automatically, validates yield bounds, and interpolates missing tenors.
- **Holiday / Weekend Fallback (`--as-of`)**: Automatically resolves non-trading dates to the nearest prior trading day.
- **Continuous Curve Fitting**: Supports linear, natural cubic spline, and **4-parameter Nelson-Siegel** curve fitting.
- **Fixed-Income Shift Analytics**: Computes tenor-by-tenor basis-point (`bps`) shifts, `1s10s` and `2s10s` term spreads, `2s5s10s` butterfly curvature, and market regime classification (*Bull/Bear Steepening/Flattening*).
- **Two-Panel Diagnostic Visualization**: Renders both the continuous/discrete yield curves (`%`) and the basis-point shift bar chart (`bps`), with support for headless PNG/SVG/PDF export.

## Project Structure

```text
yield-curve-analysis/
├── README.md
├── pyproject.toml
├── requirements.txt
├── gsec_yields.csv
├── src/
│   ├── __init__.py
│   ├── exceptions.py
│   ├── yield_data.py
│   ├── curve_builder.py
│   ├── visualizer.py
│   └── main.py
└── tests/
    └── test_yield_curve.py
```

## Installation & Usage

Install dependencies inside a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the default analysis (works from either the project root or `src/`):

```bash
python3 src/main.py
```

### CLI Options

Compare custom dates, fit a Nelson-Siegel curve, enable weekend/holiday fallback, and export both the chart and JSON metrics:

```bash
python3 src/main.py \
  --dates "Feb 2024=2024-02-01" "Jun 2024=2024-06-03" "Oct 2024=2024-10-01" \
  --interpolation nelson_siegel \
  --as-of \
  --output output/gsec_comparison.png \
  --export-json output/gsec_metrics.json \
  --no-show
```

### Running Tests

```bash
python3 -m unittest discover -s tests -v
```
