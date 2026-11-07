"""Indian G-Sec Yield Curve Analysis package."""

from __future__ import annotations

try:
    from .curve_builder import CurveShiftMetrics, YieldCurve, YieldCurveBuilder
    from .exceptions import (
        CurveDateNotFoundError,
        DataValidationError,
        InvalidTenorError,
        YieldCurveError,
    )
    from .visualizer import YieldCurveVisualizer
    from .yield_data import YieldMarketData, format_years_to_tenor, parse_tenor_to_years
except ImportError:  # pragma: no cover - direct script execution fallback
    from curve_builder import CurveShiftMetrics, YieldCurve, YieldCurveBuilder  # type: ignore[no-redef]
    from exceptions import (  # type: ignore[no-redef]
        CurveDateNotFoundError,
        DataValidationError,
        InvalidTenorError,
        YieldCurveError,
    )
    from visualizer import YieldCurveVisualizer  # type: ignore[no-redef]
    from yield_data import (  # type: ignore[no-redef]
        YieldMarketData,
        format_years_to_tenor,
        parse_tenor_to_years,
    )

__all__ = [
    "CurveDateNotFoundError",
    "CurveShiftMetrics",
    "DataValidationError",
    "InvalidTenorError",
    "YieldCurve",
    "YieldCurveBuilder",
    "YieldCurveError",
    "YieldCurveVisualizer",
    "YieldMarketData",
    "format_years_to_tenor",
    "parse_tenor_to_years",
]
