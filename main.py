import numpy as np

from drawer import _merge_phi_delta, draw_unit_circle_interactive
from recurrences import *

if __name__ == "__main__":
    phi, delta = _merge_phi_delta(phi_p, delta_p, phi_m, delta_m)
    draw_unit_circle_interactive(theta, phi=phi, delta=delta)
