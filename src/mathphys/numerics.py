"""Basic numerical methods for Mathematical Physics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def integrate_trapezoid(f: ArrayLike, x: ArrayLike) -> float:
    """Compute the definite integral of f over x using the trapezoidal rule.

    Parameters
    ----------
    f : array-like
        Function values evaluated at points x.
    x : array-like
        Abscissa (must be monotonically increasing).

    Returns
    -------
    float
        Approximate value of the integral.
    """
    f_arr = np.asarray(f, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    return float(np.trapezoid(f_arr, x_arr))


def finite_difference(f: ArrayLike, x: ArrayLike, order: int = 1) -> np.ndarray:
    """Estimate derivatives of f at points x using central finite differences.

    Parameters
    ----------
    f : array-like
        Function values.
    x : array-like
        Points at which f is evaluated (uniform or non-uniform).
    order : int
        Derivative order (1 or 2).

    Returns
    -------
    np.ndarray
        Estimated derivative values (same shape as x, boundary via forward/backward diff).
    """
    f_arr = np.asarray(f, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if order == 1:
        return np.gradient(f_arr, x_arr)
    if order == 2:
        return np.gradient(np.gradient(f_arr, x_arr), x_arr)
    raise ValueError(f"Unsupported derivative order: {order}. Use 1 or 2.")
