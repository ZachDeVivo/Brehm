import numpy as np

TWO_PI = 2 * np.pi


def dist_p(a: float, b: float) -> float:
    """CCW distance from point a to point b on the unit circle."""
    return (b - a) % TWO_PI


def dist_m(a: float, b: float) -> float:
    """CW distance from point a to point b on the unit circle."""
    return (a - b) % TWO_PI
