from yield_data import YieldMarketData
from curve_builder import YieldCurveBuilder
from visualizer import YieldCurveVisualizer


CSV_FILE = "gsec_yields.csv"

DATES = {
    "Feb 2024": "2024-02-01",
    "Jun 2024": "2024-06-03"
}


def main():
    market_data = YieldMarketData(CSV_FILE)
    builder = YieldCurveBuilder(market_data)
    visualizer = YieldCurveVisualizer()

    curves = {}
    for label, date in DATES.items():
        curves[label] = builder.build_curve(date)

    visualizer.plot_curves(curves)


if __name__ == "__main__":
    main()
