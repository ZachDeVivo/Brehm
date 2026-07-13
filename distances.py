import numpy as np
import math

TWO_PI = 2 * np.pi


def dist_p(a: float, b: float) -> float:
    """CCW distance from point a to point b on the unit circle."""
    if (b == a):
        return TWO_PI
    return (b - a) % TWO_PI


def dist_m(a: float, b: float) -> float:
    """CW distance from point a to point b on the unit circle."""
    if (b == a):
        return TWO_PI
    return (a - b) % TWO_PI

def get_new_magnitude(start_angle, old_end_angle, new_end_angle, old_magnitude):
    return (old_magnitude)*math.sin(start_angle - old_end_angle)/ (old_magnitude * math.sin(new_end_angle - old_end_angle) + math.sin(start_angle - new_end_angle))