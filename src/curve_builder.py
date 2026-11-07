"""Yield curve construction, interpolation (Spline & Nelson-Siegel), and shift analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from .exceptions import CurveDateNotFoundError, DataValidationError
    from .yield_data import YieldMarketData, format_years_to_tenor, parse_tenor_to_years
except ImportError:  # pragma: no cover - direct script execution fallback
    from exceptions import CurveDateNotFoundError, DataValidationError  # type: ignore[no-redef]
    from yield_data import (  # type: ignore[no-redef]
        YieldMarketData,
        format_years_to_tenor,
        parse_tenor_to_years,
    )

InterpolationMethod = Literal["linear", "cubic", "nelson_siegel"]


def _nelson_siegel_loadings(t: np.ndarray, tau: float) -> np.ndarray:
    """Compute the [1, L1(t), L2(t)] factor loadings for the Nelson-Siegel model."""
    ratio = np.asarray(t, dtype=float) / float(tau)
    # Guard against very small ratio using Taylor expansion
    factor1 = np.where(ratio < 1e-6, 1.0 - 0.5 * ratio, (1.0 - np.exp(-ratio)) / ratio)
    factor2 = factor1 - np.exp(-ratio)
    return np.column_stack([np.ones_like(ratio), factor1, factor2])


def fit_nelson_siegel(
    maturities: np.ndarray, yields: np.ndarray
) -> dict[str, float]:
    """Fit a 4-parameter Nelson-Siegel curve via profile OLS over decay factor tau.

    Model:
        y(t) = beta0 + beta1 * ((1 - exp(-t/tau)) / (t/tau))
                     + beta2 * (((1 - exp(-t/tau)) / (t/tau)) - exp(-t/tau))
    """
    t = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)
    if len(t) < 3:
        raise DataValidationError(
            "At least 3 maturities are required for Nelson-Siegel curve fitting."
        )

    # Grid search + refinement over decay parameter tau in [0.15, 10.0] years
    tau_grid = np.concatenate([
        np.linspace(0.15, 2.5, 120),
        np.linspace(2.5, 10.0, 80),
    ])

    best_sse = float("inf")
    best_beta = np.zeros(3, dtype=float)
    best_tau = 1.5

    for tau_candidate in tau_grid:
        design = _nelson_siegel_loadings(t, float(tau_candidate))
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ beta
        sse = float(np.dot(residuals, residuals))
        if sse < best_sse:
            best_sse = sse
            best_beta = beta
            best_tau = float(tau_candidate)

    # Fine local grid around best_tau
    fine_low = max(0.05, best_tau - 0.15)
    fine_high = best_tau + 0.15
    for tau_candidate in np.linspace(fine_low, fine_high, 60):
        design = _nelson_siegel_loadings(t, float(tau_candidate))
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ beta
        sse = float(np.dot(residuals, residuals))
        if sse < best_sse:
            best_sse = sse
            best_beta = beta
            best_tau = float(tau_candidate)

    rmse_bps = float(np.sqrt(best_sse / len(t)) * 100.0)
    return {
        "beta0": float(best_beta[0]),
        "beta1": float(best_beta[1]),
        "beta2": float(best_beta[2]),
        "tau": float(best_tau),
        "rmse_bps": rmse_bps,
    }


def _evaluate_natural_cubic_spline(
    x_nodes: np.ndarray, y_nodes: np.ndarray, x_eval: np.ndarray
) -> np.ndarray:
    """Evaluate a natural cubic spline with linear extrapolation outside [x_min, x_max]."""
    n = len(x_nodes)
    if n < 3:
        return np.interp(x_eval, x_nodes, y_nodes)

    h = np.diff(x_nodes)
    # Build tridiagonal system for second derivatives M_0 ... M_{n-1} with M_0 = M_{n-1} = 0
    a = np.zeros((n, n), dtype=float)
    rhs = np.zeros(n, dtype=float)
    a[0, 0] = 1.0
    a[-1, -1] = 1.0
    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1] / 6.0
        a[i, i] = (h[i - 1] + h[i]) / 3.0
        a[i, i + 1] = h[i] / 6.0
        rhs[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (
            y_nodes[i] - y_nodes[i - 1]
        ) / h[i - 1]

    m = np.linalg.solve(a, rhs)
    x_q = np.asarray(x_eval, dtype=float)
    result = np.empty_like(x_q, dtype=float)

    # Segment indices clipped to [0, n - 2]
    idx = np.clip(np.searchsorted(x_nodes, x_q, side="right") - 1, 0, n - 2)
    x_i = x_nodes[idx]
    x_ip1 = x_nodes[idx + 1]
    h_i = h[idx]
    dx_left = x_ip1 - x_q
    dx_right = x_q - x_i

    result = (
        m[idx] * (dx_left**3) / (6.0 * h_i)
        + m[idx + 1] * (dx_right**3) / (6.0 * h_i)
        + (y_nodes[idx] - m[idx] * (h_i**2) / 6.0) * (dx_left / h_i)
        + (y_nodes[idx + 1] - m[idx + 1] * (h_i**2) / 6.0) * (dx_right / h_i)
    )
    return result


@dataclass(frozen=True)
class YieldCurve:
    """Immutable representation of a single-date G-Sec yield curve."""

    date: pd.Timestamp
    requested_date: pd.Timestamp
    tenors: tuple[str, ...]
    maturities: np.ndarray
    yields: np.ndarray
    interpolation_method: InterpolationMethod = "cubic"
    label: str = ""
    ns_params: dict[str, float] | None = field(default=None)

    @property
    def index(self) -> pd.Index:
        """Backward-compatible tenor index matching pd.Series.index."""
        return pd.Index(self.tenors, name="Tenor")

    @property
    def values(self) -> np.ndarray:
        """Backward-compatible yield values array matching pd.Series.values."""
        return self.yields.copy()

    @property
    def empty(self) -> bool:
        """Return True if the curve contains no points."""
        return len(self.yields) == 0

    @property
    def series(self) -> pd.Series:
        """Return the discrete yield curve as a pandas Series indexed by tenor label."""
        name = self.label or self.date.strftime("%Y-%m-%d")
        return pd.Series(self.yields, index=self.index, name=name, dtype=float)

    def __len__(self) -> int:
        return len(self.yields)

    def __getitem__(self, item: str | int) -> float:
        if isinstance(item, int):
            return float(self.yields[item])
        if item in self.tenors:
            pos = self.tenors.index(item)
            return float(self.yields[pos])
        years = parse_tenor_to_years(item)
        return float(self.evaluate(years))

    def evaluate(
        self,
        target_maturities: float | list[float] | np.ndarray,
        method: InterpolationMethod | None = None,
    ) -> float | np.ndarray:
        """Evaluate the yield (%) at arbitrary maturity/maturities (in years)."""
        active_method = method or self.interpolation_method
        is_scalar = np.ndim(target_maturities) == 0
        x_eval = np.atleast_1d(np.asarray(target_maturities, dtype=float))
        if np.any(x_eval <= 0):
            raise DataValidationError("Target maturities in years must be > 0.")

        if active_method == "linear" or len(self.maturities) < 3:
            y_eval = np.interp(x_eval, self.maturities, self.yields)
        elif active_method == "cubic":
            y_eval = _evaluate_natural_cubic_spline(
                self.maturities, self.yields, x_eval
            )
        elif active_method == "nelson_siegel":
            params = self.ns_params or fit_nelson_siegel(self.maturities, self.yields)
            design = _nelson_siegel_loadings(x_eval, params["tau"])
            beta = np.array([params["beta0"], params["beta1"], params["beta2"]])
            y_eval = design @ beta
        else:
            raise DataValidationError(f"Unknown interpolation method: {active_method!r}")

        if is_scalar:
            return float(y_eval[0])
        return y_eval

    def term_spread_bps(
        self, short_years: float = 1.0, long_years: float = 10.0
    ) -> float:
        """Compute term spread (long_yield - short_yield) in basis points (bps)."""
        y_short = float(self.evaluate(short_years))
        y_long = float(self.evaluate(long_years))
        return (y_long - y_short) * 100.0

    def butterfly_bps(
        self,
        short_years: float = 2.0,
        mid_years: float = 5.0,
        long_years: float = 10.0,
    ) -> float:
        """Compute 2*Y(mid) - (Y(short) + Y(long)) butterfly curvature in basis points."""
        y_short = float(self.evaluate(short_years))
        y_mid = float(self.evaluate(mid_years))
        y_long = float(self.evaluate(long_years))
        return (2.0 * y_mid - (y_short + y_long)) * 100.0

    def discount_factor(
        self,
        maturity_years: float,
        compounding: Literal["semi_annual", "annual", "continuous"] = "semi_annual",
    ) -> float:
        """Compute the discount factor P(0, t) for ``maturity_years``.

        Defaults to ``semi_annual`` compounding (Indian G-Sec market convention).
        """
        r = float(self.evaluate(maturity_years)) / 100.0
        t = float(maturity_years)
        if compounding == "continuous":
            return float(np.exp(-r * t))
        if compounding == "annual":
            return float((1.0 + r) ** (-t))
        if compounding == "semi_annual":
            return float((1.0 + r / 2.0) ** (-2.0 * t))
        raise DataValidationError(f"Unsupported compounding convention: {compounding!r}")

    def forward_rate(
        self,
        t1_years: float,
        t2_years: float,
    ) -> float:
        """Compute the implied annualized forward rate (%) between ``t1_years`` and ``t2_years``."""
        if t1_years <= 0 or t2_years <= t1_years:
            raise DataValidationError(
                f"Forward rate requires 0 < t1_years < t2_years, got ({t1_years}, {t2_years})."
            )
        y1 = float(self.evaluate(t1_years)) / 100.0
        y2 = float(self.evaluate(t2_years)) / 100.0
        # Continuously compounded implied forward rate converted to percentage
        fwd = (y2 * t2_years - y1 * t1_years) / (t2_years - t1_years)
        return float(fwd * 100.0)


@dataclass(frozen=True)
class CurveShiftMetrics:
    """Quantitative comparison between a base yield curve and a target yield curve."""

    base_label: str
    target_label: str
    base_date: pd.Timestamp
    target_date: pd.Timestamp
    tenors: tuple[str, ...]
    maturities: np.ndarray
    tenor_shifts_bps: np.ndarray
    parallel_shift_bps: float
    slope_1s10s_base_bps: float
    slope_1s10s_target_bps: float
    slope_1s10s_change_bps: float
    slope_2s10s_base_bps: float
    slope_2s10s_target_bps: float
    slope_2s10s_change_bps: float
    butterfly_2s5s10s_base_bps: float
    butterfly_2s5s10s_target_bps: float
    butterfly_2s5s10s_change_bps: float
    regime: str

    @property
    def shifts_series(self) -> pd.Series:
        """Return basis-point shifts indexed by tenor."""
        return pd.Series(
            self.tenor_shifts_bps,
            index=pd.Index(self.tenors, name="Tenor"),
            name=f"{self.target_label} vs {self.base_label} (bps)",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize shift metrics to a dictionary for JSON/CSV reporting."""
        return {
            "base_label": self.base_label,
            "target_label": self.target_label,
            "base_date": self.base_date.strftime("%Y-%m-%d"),
            "target_date": self.target_date.strftime("%Y-%m-%d"),
            "parallel_shift_bps": round(self.parallel_shift_bps, 2),
            "slope_1s10s_base_bps": round(self.slope_1s10s_base_bps, 2),
            "slope_1s10s_target_bps": round(self.slope_1s10s_target_bps, 2),
            "slope_1s10s_change_bps": round(self.slope_1s10s_change_bps, 2),
            "slope_2s10s_base_bps": round(self.slope_2s10s_base_bps, 2),
            "slope_2s10s_target_bps": round(self.slope_2s10s_target_bps, 2),
            "slope_2s10s_change_bps": round(self.slope_2s10s_change_bps, 2),
            "butterfly_2s5s10s_base_bps": round(self.butterfly_2s5s10s_base_bps, 2),
            "butterfly_2s5s10s_target_bps": round(self.butterfly_2s5s10s_target_bps, 2),
            "butterfly_2s5s10s_change_bps": round(self.butterfly_2s5s10s_change_bps, 2),
            "regime": self.regime,
            "tenor_shifts_bps": {
                t: round(float(val), 2)
                for t, val in zip(self.tenors, self.tenor_shifts_bps)
            },
        }


class YieldCurveBuilder:
    """Constructs, interpolates, and compares yield curves for selected dates."""

    def __init__(
        self,
        market_data: YieldMarketData,
        *,
        interpolation_method: InterpolationMethod = "cubic",
        as_of: bool = False,
    ) -> None:
        self.market_data = market_data
        self.interpolation_method: InterpolationMethod = interpolation_method
        self.as_of = as_of

    def build_curve(
        self,
        date: str | pd.Timestamp,
        *,
        label: str | None = None,
        as_of: bool | None = None,
        interpolation_method: InterpolationMethod | None = None,
    ) -> YieldCurve:
        """Build a validated, maturity-sorted :class:`YieldCurve` for ``date``."""
        use_as_of = self.as_of if as_of is None else as_of
        method = interpolation_method or self.interpolation_method

        requested_ts = pd.to_datetime(date, errors="raise").normalize()
        resolved_ts = self.market_data.resolve_date(requested_ts, as_of=use_as_of)
        row_df = self.market_data.get_curve_for_date(
            resolved_ts, as_of=False, raise_if_missing=True
        )
        if row_df.empty:
            raise CurveDateNotFoundError(f"No curve data available for date {date!r}")

        # Extract single row deterministically and drop any residual NaNs
        series = row_df.iloc[0].dropna()
        if len(series) < 2:
            raise DataValidationError(
                f"Date '{resolved_ts.strftime('%Y-%m-%d')}' has fewer than 2 valid yield points."
            )

        # Parse and sort maturities ascending
        tenor_pairs = [
            (str(col), parse_tenor_to_years(str(col)), float(val))
            for col, val in series.items()
        ]
        tenor_pairs.sort(key=lambda item: item[1])

        tenors = tuple(item[0] for item in tenor_pairs)
        maturities = np.array([item[1] for item in tenor_pairs], dtype=float)
        yields = np.array([item[2] for item in tenor_pairs], dtype=float)

        ns_params: dict[str, float] | None = None
        if method == "nelson_siegel" and len(maturities) >= 3:
            ns_params = fit_nelson_siegel(maturities, yields)

        resolved_label = label or resolved_ts.strftime("%Y-%m-%d")
        return YieldCurve(
            date=resolved_ts,
            requested_date=requested_ts,
            tenors=tenors,
            maturities=maturities,
            yields=yields,
            interpolation_method=method,
            label=resolved_label,
            ns_params=ns_params,
        )

    @staticmethod
    def coerce_curve(
        curve_or_series: YieldCurve | pd.Series, label: str = ""
    ) -> YieldCurve:
        """Convert a legacy ``pd.Series`` or :class:`YieldCurve` into a :class:`YieldCurve`."""
        if isinstance(curve_or_series, YieldCurve):
            return curve_or_series

        if isinstance(curve_or_series, pd.DataFrame):
            if curve_or_series.empty:
                raise DataValidationError(f"Curve '{label}' is an empty DataFrame.")
            series = curve_or_series.iloc[0].dropna()
        elif isinstance(curve_or_series, pd.Series):
            series = curve_or_series.dropna()
        else:
            raise DataValidationError(
                f"Expected YieldCurve or pd.Series, got {type(curve_or_series).__name__}"
            )

        if series.empty:
            raise DataValidationError(f"Curve '{label}' contains no valid yield points.")

        pairs = [
            (str(col), parse_tenor_to_years(str(col)), float(val))
            for col, val in series.items()
        ]
        pairs.sort(key=lambda item: item[1])
        now_ts = pd.Timestamp.now().normalize()
        return YieldCurve(
            date=now_ts,
            requested_date=now_ts,
            tenors=tuple(p[0] for p in pairs),
            maturities=np.array([p[1] for p in pairs], dtype=float),
            yields=np.array([p[2] for p in pairs], dtype=float),
            interpolation_method="cubic" if len(pairs) >= 3 else "linear",
            label=label or str(series.name or "Curve"),
        )

    @classmethod
    def compare_curves(
        cls,
        base_curve: YieldCurve | pd.Series,
        target_curve: YieldCurve | pd.Series,
        *,
        base_label: str | None = None,
        target_label: str | None = None,
    ) -> CurveShiftMetrics:
        """Compute basis-point shifts, slope changes, butterfly curvature, and shift regime."""
        base = cls.coerce_curve(base_curve, label=base_label or "Base")
        target = cls.coerce_curve(target_curve, label=target_label or "Target")

        b_label = base_label or base.label or base.date.strftime("%Y-%m-%d")
        t_label = target_label or target.label or target.date.strftime("%Y-%m-%d")

        # Evaluate on common maturities if available, otherwise union within overlapping domain
        common_maturities = np.intersect1d(base.maturities, target.maturities)
        if len(common_maturities) == 0:
            low = max(float(base.maturities.min()), float(target.maturities.min()))
            high = min(float(base.maturities.max()), float(target.maturities.max()))
            common_maturities = np.linspace(low, high, 5)

        tenor_labels = tuple(format_years_to_tenor(float(m)) for m in common_maturities)
        base_vals = np.asarray(base.evaluate(common_maturities), dtype=float)
        target_vals = np.asarray(target.evaluate(common_maturities), dtype=float)

        shifts_bps = (target_vals - base_vals) * 100.0
        parallel_shift_bps = float(np.mean(shifts_bps))

        # 1s10s and 2s10s slopes (clipped to curve maturity range if needed)
        min_m = max(float(base.maturities.min()), float(target.maturities.min()))
        max_m = min(float(base.maturities.max()), float(target.maturities.max()))
        s1_short = max(1.0, min_m)
        s2_short = max(2.0, min_m)
        s_long = min(10.0, max_m)
        s_mid = min(max(5.0, min_m), max_m)

        slope_1s10s_base = base.term_spread_bps(s1_short, s_long)
        slope_1s10s_target = target.term_spread_bps(s1_short, s_long)
        slope_1s10s_change = slope_1s10s_target - slope_1s10s_base

        slope_2s10s_base = base.term_spread_bps(s2_short, s_long)
        slope_2s10s_target = target.term_spread_bps(s2_short, s_long)
        slope_2s10s_change = slope_2s10s_target - slope_2s10s_base

        bfly_base = base.butterfly_bps(s2_short, s_mid, s_long)
        bfly_target = target.butterfly_bps(s2_short, s_mid, s_long)
        bfly_change = bfly_target - bfly_base

        regime = cls._classify_regime(parallel_shift_bps, slope_1s10s_change)

        return CurveShiftMetrics(
            base_label=b_label,
            target_label=t_label,
            base_date=base.date,
            target_date=target.date,
            tenors=tenor_labels,
            maturities=common_maturities,
            tenor_shifts_bps=shifts_bps,
            parallel_shift_bps=parallel_shift_bps,
            slope_1s10s_base_bps=slope_1s10s_base,
            slope_1s10s_target_bps=slope_1s10s_target,
            slope_1s10s_change_bps=slope_1s10s_change,
            slope_2s10s_base_bps=slope_2s10s_base,
            slope_2s10s_target_bps=slope_2s10s_target,
            slope_2s10s_change_bps=slope_2s10s_change,
            butterfly_2s5s10s_base_bps=bfly_base,
            butterfly_2s5s10s_target_bps=bfly_target,
            butterfly_2s5s10s_change_bps=bfly_change,
            regime=regime,
        )

    @staticmethod
    def _classify_regime(level_change_bps: float, slope_change_bps: float) -> str:
        """Classify fixed-income curve movement into standard market regimes."""
        if abs(level_change_bps) < 1.0 and abs(slope_change_bps) < 1.0:
            return "Unchanged"
        if abs(slope_change_bps) < 1.5:
            return "Bear Parallel Shift" if level_change_bps > 0 else "Bull Parallel Shift"
        if level_change_bps >= 0 and slope_change_bps > 0:
            return "Bear Steepening"
        if level_change_bps >= 0 and slope_change_bps < 0:
            return "Bear Flattening"
        if level_change_bps < 0 and slope_change_bps > 0:
            return "Bull Steepening"
        return "Bull Flattening"
