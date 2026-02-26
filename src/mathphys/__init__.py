"""Mathematical Physics toolkit — Python layer."""

from mathphys.numerics import integrate_trapezoid, finite_difference
from mathphys.double_pendulum import DoublePendulum

__all__ = ["integrate_trapezoid", "finite_difference", "DoublePendulum"]
