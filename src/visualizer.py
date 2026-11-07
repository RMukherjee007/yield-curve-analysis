"""Visualization utilities for Indian G-Sec yield curves and basis-point shifts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib"),
)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

try:
    from .curve_builder import YieldCurve, YieldCurveBuilder
    from .exceptions import DataValidationError
except ImportError:  # pragma: no cover - direct script execution fallback
    from curve_builder import YieldCurve, YieldCurveBuilder  # type: ignore[no-redef]
    from exceptions import DataValidationError  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class YieldCurveVisualizer:
    """Plots and compares Indian G-Sec yield curves and basis-point shifts."""

    def plot_curves(
        self,
        curves: Mapping[str, YieldCurve | pd.Series],
        title: str = "Indian G-Sec Yield Curve Comparison",
        *,
        show_shifts: bool = True,
        smooth_curve: bool = True,
        save_path: str | Path | None = None,
        dpi: int = 150,
        show: bool = True,
    ) -> tuple[Figure, tuple[Axes, ...]]:
        """Plot one or more yield curves with an optional basis-point shift panel.

        Parameters
        ----------
        curves:
            Mapping of curve label to :class:`YieldCurve` or ``pd.Series``.
        title:
            Chart title for the top yield curve panel.
        show_shifts:
            If ``True`` and 2+ curves are supplied, adds a bottom subplot showing
            tenor-by-tenor shifts in basis points (bps) relative to the first curve.
        smooth_curve:
            If ``True``, plots the continuous interpolated curve alongside discrete
            market observations.
        save_path:
            Optional file path (e.g. ``output/yield_curve.png``) to save the figure.
        dpi:
            Resolution in dots per inch when saving.
        show:
            Whether to display the figure interactively when an interactive backend is active.
        """
        if not curves:
            raise DataValidationError("Cannot plot empty curves dictionary.")

        if not show and "agg" not in matplotlib.get_backend().lower():
            plt.switch_backend("Agg")

        coerced_curves: dict[str, YieldCurve] = {
            label: YieldCurveBuilder.coerce_curve(curve_obj, label=label)
            for label, curve_obj in curves.items()
        }

        has_shift_panel = show_shifts and len(coerced_curves) >= 2
        if has_shift_panel:
            fig, (ax_curve, ax_shift) = plt.subplots(
                2,
                1,
                figsize=(11, 8),
                gridspec_kw={"height_ratios": [2.2, 1.1]},
                 sharex=False,
            )
            axes: tuple[Axes, ...] = (ax_curve, ax_shift)
        else:
            fig, ax_curve = plt.subplots(1, 1, figsize=(10, 6))
            axes = (ax_curve,)

        palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]

        for idx, (label, curve) in enumerate(coerced_curves.items()):
            color = palette[idx % len(palette)]
            spread_1s10s = curve.term_spread_bps(
                max(1.0, float(curve.maturities.min())),
                min(10.0, float(curve.maturities.max())),
            )
            legend_label = f"{label} ({curve.date.strftime('%Y-%m-%d')} | 1s10s: {spread_1s10s:+.0f} bps)"

            if smooth_curve and len(curve.maturities) >= 3:
                dense_x = np.linspace(
                    float(curve.maturities.min()),
                    float(curve.maturities.max()),
                    200,
                )
                dense_y = np.asarray(curve.evaluate(dense_x), dtype=float)
                ax_curve.plot(
                    dense_x,
                    dense_y,
                    color=color,
                    linewidth=2.0,
                    label=legend_label,
                )
                ax_curve.scatter(
                    curve.maturities,
                    curve.yields,
                    color=color,
                    s=45,
                    zorder=4,
                )
            else:
                ax_curve.plot(
                    curve.maturities,
                    curve.yields,
                    marker="o",
                    color=color,
                    linewidth=2.0,
                    label=legend_label,
                )

        ax_curve.set_xlabel("Maturity (Years)")
        ax_curve.set_ylabel("Yield (%)")
        ax_curve.set_title(title, fontsize=13, fontweight="bold")
        ax_curve.legend(frameon=True)
        ax_curve.grid(alpha=0.3, linestyle="--")

        if has_shift_panel:
            ax_shift = axes[1]
            labels_list = list(coerced_curves.keys())
            base_label = labels_list[0]
            base_curve = coerced_curves[base_label]
            comparison_labels = labels_list[1:]
            num_comparisons = len(comparison_labels)
            bar_width = 0.75 / max(num_comparisons, 1)

            for c_idx, target_label in enumerate(comparison_labels):
                target_curve = coerced_curves[target_label]
                metrics = YieldCurveBuilder.compare_curves(
                    base_curve,
                    target_curve,
                    base_label=base_label,
                    target_label=target_label,
                )
                x_positions = np.arange(len(metrics.tenors), dtype=float)
                offset = (c_idx - (num_comparisons - 1) / 2.0) * bar_width
                color = palette[(c_idx + 1) % len(palette)]

                bars = ax_shift.bar(
                    x_positions + offset,
                    metrics.tenor_shifts_bps,
                    width=bar_width,
                    color=color,
                    alpha=0.82,
                    label=(
                        f"{target_label} vs {base_label} "
                        f"({metrics.regime}, Avg: {metrics.parallel_shift_bps:+.1f} bps)"
                    ),
                )
                for bar in bars:
                    height = bar.get_height()
                    va = "bottom" if height >= 0 else "top"
                    y_pad = 0.4 if height >= 0 else -0.4
                    ax_shift.annotate(
                        f"{height:+.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2.0, height + y_pad),
                        ha="center",
                        va=va,
                        fontsize=8,
                    )

                ax_shift.set_xticks(x_positions)
                ax_shift.set_xticklabels(metrics.tenors)

            ax_shift.axhline(0.0, color="black", linewidth=0.9, linestyle="-")
            ax_shift.set_xlabel("Tenor")
            ax_shift.set_ylabel("Shift (bps)")
            ax_shift.set_title(
                f"Yield Curve Shift Relative to {base_label} (Basis Points)",
                fontsize=11,
                fontweight="bold",
            )
            ax_shift.legend(frameon=True)
            ax_shift.grid(alpha=0.3, linestyle="--", axis="y")
            # Add vertical margin for bar annotations
            y_min, y_max = ax_shift.get_ylim()
            span = max(abs(y_max - y_min), 5.0)
            ax_shift.set_ylim(y_min - 0.18 * span, y_max + 0.18 * span)

        fig.tight_layout()

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            logger.info("Saved yield curve chart to '%s'.", out_path)

        backend = matplotlib.get_backend().lower()
        if show and "agg" not in backend:
            plt.show()

        return fig, axes
