# Yield Curve Analysis (Indian G-Sec)

A Python-based analysis tool to construct and compare Indian government bond yield curves across maturities.

## Overview

The yield curve is an important indicator in fixed-income markets, reflecting market expectations about future interest rates, inflation, and economic growth.  
This project compares Indian G-Sec yield curves across two different dates to visualize how expectations change over time.

The focus is on understanding curve shape and movement, not on prediction or trading strategies.

## What this project does

- Loads Indian government bond yield data across multiple maturities
- Constructs yield curves for selected dates
- Plots and compares yield curves on a single chart
- Helps analyze shifts such as steepening or flattening of the curve

## Project Structure

    yield-curve-analysis/
    ├── README.md
    ├── requirements.txt
    └── src/
        ├── main.py
        ├── yield_data.py
        ├── curve_builder.py
        └── visualizer.py

## How it works

- YieldMarketData loads yield data from a CSV file
- YieldCurveBuilder constructs yield curves for selected dates
- YieldCurveVisualizer plots and compares the curves
- main.py coordinates the overall analysis

## Data Format

The project expects a CSV file named gsec_yields.csv in the project root directory.

Example:

    Date,1Y,3Y,5Y,10Y
    2024-02-01,6.80,7.05,7.20,7.32
    2024-06-03,6.95,7.20,7.35,7.45

## Running the project

Install dependencies:

    pip install -r requirements.txt

Run the analysis:

    cd src
    python main.py

## Notes

This project is intended for educational and analytical purposes only.
