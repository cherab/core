
import os
import json
from raysect.core import Point2D

from cherab.tools.equilibrium.efit import EFITEquilibrium


def example_equilibrium():
    """
    Return a populated instance of the example equilibrium.

    .. code-block:: pycon

       >>> from cherab.tools.equilibrium import example_equilibrium
       >>> equilibrium = example_equilibrium()
    """

    directory = os.path.split(__file__)[0]
    example_file = os.path.join(directory, 'example.json')
    with open(example_file, 'r') as fh:
        eq_data = json.load(fh)

    r = eq_data['r']
    z = eq_data['z']
    psi = eq_data['psi']
    psi_axis = eq_data['psi_axis']
    psi_lcfs = eq_data['psi_lcfs']
    ac = eq_data['axis_coord']
    axis_coord = Point2D(ac[0], ac[1])
    xp = eq_data['x_points']
    x_points = [Point2D(xp[0][0], xp[0][1])]
    sp = eq_data['strike_points']
    strike_points = [Point2D(sp[0][0], sp[0][1]), Point2D(sp[1][0], sp[1][1])]
    f_profile = eq_data['f_profile']
    q_profile = eq_data['q_profile']
    b_vacuum_radius = eq_data['b_vacuum_radius']
    b_vacuum_magnitude = eq_data['b_vacuum_magnitude']
    lcfs_polygon = eq_data['lcfs_polygon']
    limiter_polygon = eq_data['limiter_polygon']
    time = eq_data['time']

    equilibrium = EFITEquilibrium(r, z, psi, psi_axis, psi_lcfs, axis_coord, x_points, strike_points,
                                  f_profile, q_profile, b_vacuum_radius, b_vacuum_magnitude,
                                  lcfs_polygon, limiter_polygon, time)

    return equilibrium
