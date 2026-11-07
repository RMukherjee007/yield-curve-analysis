"""Unit and integration tests for Indian G-Sec yield curve analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.curve_builder import YieldCurve, YieldCurveBuilder, fit_nelson_siegel
from src.exceptions import (
    CurveDateNotFoundError,
    DataValidationError,
    InvalidTenorError,
)
from src.main import main
from src.visualizer import YieldCurveVisualizer
from src.yield_data import YieldMarketData, format_years_to_tenor, parse_tenor_to_years


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = PROJECT_ROOT / "gsec_yields.csv"


class TestTenorParsing(unittest.TestCase):
    """Test parsing and formatting of Indian G-Sec and T-Bill tenors."""

    def test_standard_tenors(self) -> None:
        self.assertAlmostEqual(parse_tenor_to_years("91D"), 0.25)
        self.assertAlmostEqual(parse_tenor_to_years("182D"), 0.50)
        self.assertAlmostEqual(parse_tenor_to_years("364D"), 1.00)
        self.assertAlmostEqual(parse_tenor_to_years("3M"), 0.25)
        self.assertAlmostEqual(parse_tenor_to_years("6M"), 0.50)
        self.assertAlmostEqual(parse_tenor_to_years("1Y"), 1.0)
        self.assertAlmostEqual(parse_tenor_to_years("2.5Y"), 2.5)
        self.assertAlmostEqual(parse_tenor_to_years("10y"), 10.0)
        self.assertAlmostEqual(parse_tenor_to_years(5), 5.0)

    def test_invalid_tenors_raise(self) -> None:
        for bad in ["", "ABC", "0Y", "-5Y", "10X", None]:
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidTenorError):
                    parse_tenor_to_years(bad)  # type: ignore[arg-type]

    def test_format_years_to_tenor(self) -> None:
        self.assertEqual(format_years_to_tenor(0.25), "3M")
        self.assertEqual(format_years_to_tenor(0.5), "6M")
        self.assertEqual(format_years_to_tenor(1.0), "1Y")
        self.assertEqual(format_years_to_tenor(10.0), "10Y")
        self.assertEqual(format_years_to_tenor(2.5), "2.5Y")


class TestYieldMarketData(unittest.TestCase):
    """Test CSV loading, validation, sorting, duplicate handling, and date resolution."""

    def test_loads_default_csv_and_sorts_maturities(self) -> None:
        md = YieldMarketData(SAMPLE_CSV)
        self.assertGreaterEqual(len(md.available_dates), 2)
        maturity_vals = [md.maturities[t] for t in md.tenors]
        self.assertEqual(maturity_vals, sorted(maturity_vals))

    def test_unsorted_columns_and_missing_interpolation(self) -> None:
        tmp_csv = PROJECT_ROOT / "tests" / "_tmp_unsorted.csv"
        try:
            tmp_csv.write_text(
                "Date,10Y,1Y,5Y,6M\n"
                "2024-02-01,7.30,7.00,-,6.90\n"
                "2024-02-01,7.32,7.02,7.17,6.92\n",
                encoding="utf-8",
            )
            md = YieldMarketData(tmp_csv, missing_policy="interpolate")
            self.assertEqual(md.tenors, ["6M", "1Y", "5Y", "10Y"])
            # Duplicate date deduplicated to 1 row
            self.assertEqual(len(md.data), 1)
            row = md.get_curve_for_date("2024-02-01")
            self.assertFalse(row.isna().any().any())
            self.assertAlmostEqual(float(row["10Y"].iloc[0]), 7.32)
        finally:
            tmp_csv.unlink(missing_ok=True)

    def test_strict_duplicates_and_bounds_validation(self) -> None:
        tmp_csv = PROJECT_ROOT / "tests" / "_tmp_bad.csv"
        try:
            tmp_csv.write_text(
                "Date,1Y,10Y\n2024-02-01,7.0,7.2\n2024-02-01,7.1,7.3\n",
                encoding="utf-8",
            )
            with self.assertRaises(DataValidationError):
                YieldMarketData(tmp_csv, strict_duplicates=True)

            tmp_csv.write_text(
                "Date,1Y,10Y\n2024-02-01,-1.5,7.2\n",
                encoding="utf-8",
            )
            with self.assertRaises(DataValidationError):
                YieldMarketData(tmp_csv)
        finally:
            tmp_csv.unlink(missing_ok=True)

    def test_as_of_date_resolution(self) -> None:
        md = YieldMarketData(SAMPLE_CSV)
        # 2024-02-04 was a Sunday; should resolve back to 2024-02-01 when as_of=True
        resolved = md.resolve_date("2024-02-04", as_of=True)
        self.assertEqual(resolved, pd.Timestamp("2024-02-01"))

        with self.assertRaises(CurveDateNotFoundError):
            md.resolve_date("2024-02-04", as_of=False)


class TestYieldCurveBuilderAndAnalytics(unittest.TestCase):
    """Test curve construction, interpolation methods, and shift decomposition."""

    def setUp(self) -> None:
        self.market_data = YieldMarketData(SAMPLE_CSV)
        self.builder = YieldCurveBuilder(self.market_data)

    def test_build_curve_and_interpolation_methods(self) -> None:
        for method in ("linear", "cubic", "nelson_siegel"):
            with self.subTest(method=method):
                curve = self.builder.build_curve(
                    "2024-02-01",
                    label="Feb 2024",
                    interpolation_method=method,  # type: ignore[arg-type]
                )
                self.assertIsInstance(curve, YieldCurve)
                self.assertEqual(len(curve), len(self.market_data.tenors))
                # Evaluate at 4Y (in-between 3Y and 5Y)
                y_4y = float(curve.evaluate(4.0))
                self.assertGreater(y_4y, 6.0)
                self.assertLess(y_4y, 8.0)

    def test_nelson_siegel_fit_accuracy(self) -> None:
        curve = self.builder.build_curve(
            "2024-02-01", interpolation_method="nelson_siegel"
        )
        self.assertIsNotNone(curve.ns_params)
        assert curve.ns_params is not None
        self.assertLess(curve.ns_params["rmse_bps"], 10.0)

        params = fit_nelson_siegel(curve.maturities, curve.yields)
        self.assertIn("beta0", params)
        self.assertIn("tau", params)

    def test_compare_curves_metrics(self) -> None:
        c_feb = self.builder.build_curve("2024-02-01", label="Feb 2024")
        c_jun = self.builder.build_curve("2024-06-03", label="Jun 2024")
        metrics = self.builder.compare_curves(c_feb, c_jun)
        self.assertEqual(metrics.base_label, "Feb 2024")
        self.assertEqual(metrics.target_label, "Jun 2024")
        # Jun 2024 yields are lower than Feb 2024 yields
        self.assertLess(metrics.parallel_shift_bps, 0.0)
        summary_dict = metrics.to_dict()
        self.assertIn("regime", summary_dict)
        self.assertIn("tenor_shifts_bps", summary_dict)

    def test_discount_factor_and_forward_rate(self) -> None:
        curve = self.builder.build_curve("2024-02-01", label="Feb 2024")
        df_1y = curve.discount_factor(1.0, compounding="semi_annual")
        df_10y = curve.discount_factor(10.0, compounding="semi_annual")
        self.assertGreater(df_1y, df_10y)
        self.assertLess(df_1y, 1.0)
        fwd_1y_5y = curve.forward_rate(1.0, 5.0)
        self.assertGreater(fwd_1y_5y, 5.0)
        self.assertLess(fwd_1y_5y, 9.0)


class TestVisualizerAndCLI(unittest.TestCase):
    """Test headless plotting and end-to-end CLI execution."""

    def test_visualizer_plots_and_saves_png(self) -> None:
        md = YieldMarketData(SAMPLE_CSV)
        builder = YieldCurveBuilder(md)
        curves = {
            "Feb 2024": builder.build_curve("2024-02-01", label="Feb 2024"),
            "Jun 2024": builder.build_curve("2024-06-03", label="Jun 2024"),
        }
        out_png = PROJECT_ROOT / "tests" / "_tmp_plot.png"
        try:
            visualizer = YieldCurveVisualizer()
            fig, axes = visualizer.plot_curves(
                curves, save_path=out_png, show=False
            )
            self.assertEqual(len(axes), 2)
            self.assertTrue(out_png.is_file())
            plt.close(fig)
        finally:
            out_png.unlink(missing_ok=True)

    def test_visualizer_accepts_legacy_pandas_series(self) -> None:
        legacy_curves = {
            "Curve A": pd.Series([6.8, 7.0, 7.2], index=["1Y", "5Y", "10Y"]),
            "Curve B": pd.Series([6.9, 7.1, 7.25], index=["1Y", "5Y", "10Y"]),
        }
        visualizer = YieldCurveVisualizer()
        fig, axes = visualizer.plot_curves(legacy_curves, show=False)
        self.assertEqual(len(axes), 2)
        plt.close(fig)

    def test_cli_main_execution(self) -> None:
        out_json = PROJECT_ROOT / "tests" / "_tmp_report.json"
        try:
            curves = main([
                "--csv",
                str(SAMPLE_CSV),
                "--dates",
                "Feb 2024=2024-02-01",
                "Jun 2024=2024-06-03",
                "--interpolation",
                "nelson_siegel",
                "--export-json",
                str(out_json),
                "--no-show",
            ])
            self.assertIn("Feb 2024", curves)
            self.assertIn("Jun 2024", curves)
            self.assertTrue(out_json.is_file())
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["interpolation_method"], "nelson_siegel")
            self.assertEqual(len(payload["comparisons"]), 1)
        finally:
            out_json.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
