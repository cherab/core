from cherab.core.math import Interpolate1DCubic
from cherab.core.atomic import neon, hydrogen

from cherab.tools.plasmas.ionisationbalance import match_bulk_element_density, from_element_density

from cherab.openadas import OpenADAS

import numpy as np

def doubleparabola(r, Centre, Edge, p, q):
        return (Centre - Edge) * np.power((1 - np.power((r - r.min()) / (r.max() - r.min()), p)), q) +Edge

psin_1d = np.linspace(0, 1, 50, endpoint=True)
t_e = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 3000, 1, 2, 2))
n_e = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 6e19, 1e18, 2, 2))

t_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 1500, 0, 2, 2))
n_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 2000, 0, 2, 2))

t_ne = Interpolate1DCubic(psin_1d, doubleparabola(psin_1d, 1500, 0, 2, 2))

adas = OpenADAS(permit_extrapolation=True)


balance_neon = from_element_density(adas, neon, 1e17, n_e(0.95), t_e(0.95), hydrogen, 1e16)
bulk_density = match_bulk_element_density(adas, hydrogen, balance_neon, n_e(0.95), t_e(0.95))