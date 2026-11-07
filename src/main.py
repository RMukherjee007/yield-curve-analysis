"""CLI entry point for Indian G-Sec yield curve analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

try:
    from .curve_builder import (
        CurveShiftMetrics,
        InterpolationMethod,
        YieldCurve,
        YieldCurveBuilder,
    )
    from .visualizer import YieldCurveVisualizer
    from .yield_data import YieldMarketData
except ImportError:  # pragma: no cover - direct script execution fallback
    from curve_builder import (  # type: ignore[no-redef]
        CurveShiftMetrics,
        InterpolationMethod,
        YieldCurve,
        YieldCurveBuilder,
    )
    from visualizer import YieldCurveVisualizer  # type: ignore[no-redef]
    from yield_data import YieldMarketData  # type: ignore[no-redef]

DEFAULT_CSV_FILE = Path(__file__).resolve().parent.parent / "gsec_yields.csv"
CSV_FILE = str(DEFAULT_CSV_FILE)

DATES: dict[str, str] = {
    "Feb 2024": "2024-02-01",
    "Jun 2024": "2024-06-03",
}


def _parse_dates_arg(date_args: Sequence[str] | None) -> dict[str, str]:
    """Parse CLI ``--dates`` entries of the form ``'Label=YYYY-MM-DD'`` or ``'YYYY-MM-DD'``."""
    if not date_args:
        return dict(DATES)

    parsed: dict[str, str] = {}
    for entry in date_args:
        if "=" in entry:
            label, date_str = entry.split("=", 1)
            parsed[label.strip()] = date_str.strip()
        else:
            date_str = entry.strip()
            parsed[date_str] = date_str
    return parsed


def format_analytics_report(
    curves: dict[str, YieldCurve],
    shift_reports: list[CurveShiftMetrics],
) -> str:
    """Format a human-readable summary of curve levels, slopes, and basis-point shifts."""
    lines: list[str] = [
        "=" * 74,
        "INDIAN G-SEC YIELD CURVE ANALYTICS SUMMARY",
        "=" * 74,
        f"{'Label':<14} {'Trading Date':<14} {'1Y (%)':>8} {'10Y (%)':>9} {'1s10s (bps)':>13} {'2s5s10s (bps)':>14}",
        "-" * 74,
    ]
    for label, curve in curves.items():
        min_m = float(curve.maturities.min())
        max_m = float(curve.maturities.max())
        y_1y = float(curve.evaluate(max(1.0, min_m)))
        y_10y = float(curve.evaluate(min(10.0, max_m)))
        spread_1s10s = curve.term_spread_bps(max(1.0, min_m), min(10.0, max_m))
        bfly = curve.butterfly_bps(
            max(2.0, min_m),
            min(max(5.0, min_m), max_m),
            min(10.0, max_m),
        )
        lines.append(
            f"{label:<14} {curve.date.strftime('%Y-%m-%d'):<14} "
            f"{y_1y:>8.2f} {y_10y:>9.2f} {spread_1s10s:>+13.1f} {bfly:>+14.1f}"
        )

    if shift_reports:
        lines.extend([
            "",
            "-" * 74,
            "CURVE SHIFT DECOMPOSITION (vs. Base Curve)",
            "-" * 74,
        ])
        for report in shift_reports:
            lines.append(
                f"* {report.target_label} vs {report.base_label}: "
                f"Regime = {report.regime} | "
                f"Parallel Shift = {report.parallel_shift_bps:+.1f} bps | "
                f"1s10s Slope Change = {report.slope_1s10s_change_bps:+.1f} bps | "
                f"2s5s10s Butterfly Change = {report.butterfly_2s5s10s_change_bps:+.1f} bps"
            )
    lines.append("=" * 74)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Construct, interpolate, and compare Indian G-Sec yield curves."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=CSV_FILE,
        help="Path to G-Sec yields CSV file (default: project root gsec_yields.csv).",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        help="Dates to compare in 'Label=YYYY-MM-DD' or 'YYYY-MM-DD' format.",
    )
    parser.add_argument(
        "--interpolation",
        choices=["linear", "cubic", "nelson_siegel"],
        default="cubic",
        help="Curve interpolation/fitting method (default: cubic).",
    )
    parser.add_argument(
        "--as-of",
        action="store_true",
        help="Resolve weekend/holiday dates to the nearest prior trading date.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to save the plotted figure (e.g. output/gsec_curves.png).",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional file path to export shift analytics as JSON.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive matplotlib window.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, YieldCurve]:
    """Run the end-to-end yield curve construction, comparison, and visualization pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    dates_map = _parse_dates_arg(args.dates)
    interpolation: InterpolationMethod = args.interpolation

    market_data = YieldMarketData(args.csv)
    builder = YieldCurveBuilder(
        market_data,
        interpolation_method=interpolation,
        as_of=args.as_of,
    )
    visualizer = YieldCurveVisualizer()

    curves: dict[str, YieldCurve] = {}
    for label, date_str in dates_map.items():
        curves[label] = builder.build_curve(date_str, label=label)

    labels_list = list(curves.keys())
    shift_reports: list[CurveShiftMetrics] = []
    if len(labels_list) >= 2:
        base_label = labels_list[0]
        base_curve = curves[base_label]
        for target_label in labels_list[1:]:
            shift_reports.append(
                builder.compare_curves(
                    base_curve,
                    curves[target_label],
                    base_label=base_label,
                    target_label=target_label,
                )
            )

    report_text = format_analytics_report(curves, shift_reports)
    print(report_text)

    if args.export_json:
        json_path = Path(args.export_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "interpolation_method": interpolation,
            "curves": {
                lbl: {
                    "date": c.date.strftime("%Y-%m-%d"),
                    "tenors": list(c.tenors),
                    "yields": [round(float(v), 4) for v in c.yields],
                    "ns_params": c.ns_params,
                }
                for lbl, c in curves.items()
            },
            "comparisons": [r.to_dict() for r in shift_reports],
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    visualizer.plot_curves(
        curves,
        save_path=args.output,
        show=not args.no_show,
    )
    return curves


if __name__ == "__main__":
    main()
