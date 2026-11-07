"""Market data ingestion, tenor parsing, and validation for G-Sec yields."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

try:
    from .exceptions import (
        CurveDateNotFoundError,
        DataValidationError,
        InvalidTenorError,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from exceptions import (  # type: ignore[no-redef]
        CurveDateNotFoundError,
        DataValidationError,
        InvalidTenorError,
    )

logger = logging.getLogger(__name__)

_TENOR_PATTERN = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([DWMY])\s*$", re.IGNORECASE)
MissingPolicy = Literal["interpolate", "drop", "raise", "keep"]


def parse_tenor_to_years(tenor: str | int | float) -> float:
    """Convert a tenor label (e.g. '91D', '6M', '1Y', '10Y') into year fractions.

    Standard Indian T-Bill tenors (91D, 182D, 364D) are mapped to 0.25, 0.5, and 1.0
    years respectively; arbitrary day counts use a 365-day year convention.
    """
    if isinstance(tenor, (int, float)) and not isinstance(tenor, bool):
        years = float(tenor)
        if np.isnan(years) or years <= 0:
            raise InvalidTenorError(f"Tenor must be positive, got: {tenor!r}")
        return years

    if not isinstance(tenor, str):
        raise InvalidTenorError(f"Unsupported tenor type {type(tenor).__name__}: {tenor!r}")

    match = _TENOR_PATTERN.match(tenor)
    if not match:
        raise InvalidTenorError(
            f"Invalid tenor format {tenor!r}. Expected '<number><D|W|M|Y>' (e.g. '91D', '6M', '10Y')."
        )

    value = float(match.group(1))
    unit = match.group(2).upper()
    if value <= 0:
        raise InvalidTenorError(f"Tenor must be strictly positive, got: {tenor!r}")

    if unit == "Y":
        return value
    if unit == "M":
        return value / 12.0
    if unit == "W":
        return (value * 7.0) / 365.0
    if unit == "D":
        # Standard Indian Treasury Bill conventions (91D = 1/4 yr, 182D = 1/2 yr, 364D = 1 yr)
        if int(value) == 91 and value == 91.0:
            return 0.25
        if int(value) == 182 and value == 182.0:
            return 0.50
        if int(value) == 364 and value == 364.0:
            return 1.00
        return value / 365.0

    raise InvalidTenorError(f"Unsupported tenor unit {unit!r} in {tenor!r}")


def format_years_to_tenor(years: float) -> str:
    """Format a maturity in years back into a canonical tenor label."""
    if years <= 0:
        raise InvalidTenorError(f"Maturity years must be positive, got {years}")
    months = years * 12.0
    if years < 1.0 and abs(months - round(months)) < 1e-6:
        return f"{int(round(months))}M"
    if abs(years - round(years)) < 1e-6:
        return f"{int(round(years))}Y"
    return f"{years:g}Y"


class YieldMarketData:
    """Loads, validates, and stores G-Sec yield data across maturities."""

    def __init__(
        self,
        csv_file: str | Path,
        *,
        missing_policy: MissingPolicy = "interpolate",
        min_yield: float = 0.0,
        max_yield: float = 25.0,
        strict_duplicates: bool = False,
    ) -> None:
        self.csv_path = self._resolve_csv_path(csv_file)
        self.missing_policy: MissingPolicy = missing_policy
        self.min_yield = min_yield
        self.max_yield = max_yield
        self.strict_duplicates = strict_duplicates
        self.maturities: dict[str, float] = {}
        self.tenors: list[str] = []
        self.data = self._load_data(self.csv_path)

    @staticmethod
    def _resolve_csv_path(csv_file: str | Path) -> Path:
        path = Path(csv_file)
        if path.is_file():
            return path.resolve()

        # Fallback: check project root relative to src/ directory
        project_root_candidate = Path(__file__).resolve().parent.parent / path
        if project_root_candidate.is_file():
            return project_root_candidate.resolve()

        raise FileNotFoundError(
            f"Yield CSV file not found at '{path}' or '{project_root_candidate}'."
        )

    def _load_data(self, csv_file: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_file)
        except Exception as exc:
            raise DataValidationError(f"Failed to read CSV file '{csv_file}': {exc}") from exc

        if df.empty:
            raise DataValidationError(f"CSV file '{csv_file}' contains no rows.")

        # Normalize column whitespace
        df.columns = [str(col).strip() for col in df.columns]
        if "Date" not in df.columns:
            raise DataValidationError(
                f"CSV file '{csv_file}' must contain a 'Date' column. Found: {list(df.columns)}"
            )

        raw_tenor_cols = [col for col in df.columns if col != "Date"]
        if not raw_tenor_cols:
            raise DataValidationError(
                f"CSV file '{csv_file}' must contain at least one maturity column."
            )

        # Parse and sort maturities ascending
        parsed_tenors: list[tuple[str, float]] = []
        seen_maturities: dict[float, str] = {}
        for col in raw_tenor_cols:
            maturity_years = round(parse_tenor_to_years(col), 8)
            if maturity_years in seen_maturities:
                raise DataValidationError(
                    f"Duplicate maturity detected: '{col}' and '{seen_maturities[maturity_years]}' "
                    f"both map to {maturity_years} years."
                )
            seen_maturities[maturity_years] = col
            parsed_tenors.append((col, maturity_years))

        parsed_tenors.sort(key=lambda item: item[1])
        self.tenors = [col for col, _ in parsed_tenors]
        self.maturities = dict(parsed_tenors)

        # Parse dates
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="raise").dt.normalize()
        except Exception as exc:
            raise DataValidationError(f"Invalid date values in 'Date' column: {exc}") from exc

        if df["Date"].isna().any():
            raise DataValidationError("Found null/missing values in 'Date' column.")

        # Order columns as ['Date', *sorted_tenors] and sort chronologically
        df = df[["Date", *self.tenors]].sort_values("Date").reset_index(drop=True)

        # Coerce yield columns to float
        for col in self.tenors:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle duplicate dates
        dup_mask = df.duplicated(subset=["Date"], keep=False)
        if dup_mask.any():
            dup_dates = df.loc[dup_mask, "Date"].dt.strftime("%Y-%m-%d").unique().tolist()
            if self.strict_duplicates:
                raise DataValidationError(f"Duplicate dates found in dataset: {dup_dates}")
            logger.warning(
                "Duplicate dates %s detected in '%s'; keeping last entry per date.",
                dup_dates,
                csv_file.name,
            )
            df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        # Validate plausible yield ranges on non-NaN entries
        yield_values = df[self.tenors]
        out_of_bounds = (yield_values < self.min_yield) | (yield_values > self.max_yield)
        if out_of_bounds.any().any():
            bad_locs = out_of_bounds.stack()
            bad_entries = bad_locs[bad_locs].index.tolist()
            raise DataValidationError(
                f"Yield values outside plausible bounds [{self.min_yield}%, {self.max_yield}%] "
                f"at (row, tenor): {bad_entries[:5]}"
            )

        # Handle missing values according to policy
        if df[self.tenors].isna().any().any():
            if self.missing_policy == "raise":
                raise DataValidationError("Missing/NaN yield values found in dataset.")
            if self.missing_policy == "drop":
                df = df.dropna(subset=self.tenors, how="any").reset_index(drop=True)
            elif self.missing_policy == "interpolate":
                # Interpolate across maturity (in years) row-wise
                maturity_index = [self.maturities[c] for c in self.tenors]
                numeric_block = df[self.tenors].copy()
                numeric_block.columns = maturity_index
                numeric_block = numeric_block.interpolate(
                    method="index", axis=1, limit_direction="both"
                )
                numeric_block.columns = self.tenors
                df[self.tenors] = numeric_block

        return df

    @property
    def available_dates(self) -> list[pd.Timestamp]:
        """Return sorted list of available dates in the dataset."""
        return list(self.data["Date"])

    def resolve_date(self, date: str | pd.Timestamp, *, as_of: bool = False) -> pd.Timestamp:
        """Resolve a requested date to an available market trading date.

        If ``as_of=True``, dates falling on weekends or holidays resolve to the most
        recent prior trading date on or before ``date``.
        """
        try:
            target = pd.to_datetime(date, errors="raise").normalize()
        except Exception as exc:
            raise CurveDateNotFoundError(f"Invalid date string {date!r}: {exc}") from exc

        exact_match = self.data[self.data["Date"] == target]
        if not exact_match.empty:
            return pd.Timestamp(exact_match["Date"].iloc[0])

        if as_of:
            prior = self.data[self.data["Date"] <= target]
            if not prior.empty:
                resolved = pd.Timestamp(prior["Date"].iloc[-1])
                logger.info(
                    "Requested date %s not in dataset; using prior trading date %s.",
                    target.strftime("%Y-%m-%d"),
                    resolved.strftime("%Y-%m-%d"),
                )
                return resolved

        available_preview = [d.strftime("%Y-%m-%d") for d in self.available_dates[:10]]
        raise CurveDateNotFoundError(
            f"No yield curve data found for date '{target.strftime('%Y-%m-%d')}'"
            f"{' (or prior)' if as_of else ''}. "
            f"Available dates include: {available_preview}"
        )

    def get_curve_for_date(
        self,
        date: str | pd.Timestamp,
        *,
        as_of: bool = False,
        raise_if_missing: bool = False,
    ) -> pd.DataFrame:
        """Return a 1-row DataFrame of yields across sorted tenors for ``date``.

        For backward compatibility, returns an empty DataFrame when ``raise_if_missing=False``
        and the date is not found. Set ``raise_if_missing=True`` to raise
        :class:`CurveDateNotFoundError`.
        """
        try:
            resolved_date = self.resolve_date(date, as_of=as_of)
        except CurveDateNotFoundError:
            if raise_if_missing:
                raise
            return pd.DataFrame(columns=self.tenors, dtype=float)

        row = self.data[self.data["Date"] == resolved_date]
        return row.drop(columns=["Date"]).reset_index(drop=True)
