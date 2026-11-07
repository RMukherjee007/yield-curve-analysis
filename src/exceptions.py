"""Custom exceptions for Indian G-Sec yield curve analysis."""

from __future__ import annotations


class YieldCurveError(Exception):
    """Base exception for all yield curve domain errors."""


class DataValidationError(YieldCurveError):
    """Raised when market yield data fails schema or domain validation."""


class InvalidTenorError(DataValidationError):
    """Raised when a maturity/tenor column label cannot be parsed into years."""


class CurveDateNotFoundError(YieldCurveError):
    """Raised when a requested date is not available in the market dataset."""
