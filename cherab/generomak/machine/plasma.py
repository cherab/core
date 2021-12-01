import os
import json

from cherab.core.atomic.elements import hydrogen, carbon
from cherab.core.utility import RecursiveDict


def load_edge_profiles():
    """
    Loads Generomak edge plasma profiles

    Return a single dictionary with available edge and plasma species temperature and 
    density profiles. The profiles are saved on a 2D triangular mesh.

    :return: dictionary with mesh, electron and plasma composition profiles

    .. code-block:: pycon
       >>> # This example shows how to load data and create 2D edge interpolators
       >>>
       >>> from raysect.core.math.function.float.function2d.interpolate import Discrete2DMesh
       >>>
       >>>
       >>> data = load_edge_profiles()
       >>> 
       >>> # create electron temperature 2D mesh interpolator
       >>> te = Discrete2DMesh(data["mesh"]["vertex_coords"],
                               data["mesh"]["triangles"],
                               data["electron"]["temperature"], limit=False)

       >>> # create hydrogen 0+ density 2D mesh interpolator
       >>> n_h0 = Discrete2DMesh.instance(te, data["composition"]["hydrogen"][0]["temperature"])
    """
    profiles_dir = os.path.join(os.path.dirname(__file__), "data/plasma/edge")

    edge_data = RecursiveDict()
    path = os.path.join(profiles_dir, "mesh.json")
    with open(path, "r") as fhl:
        edge_data["mesh"] = json.load(fhl)

    path = os.path.join(profiles_dir, "electrons.json")
    with open(path, "r") as fhl:
        edge_data["electron"] = json.load(fhl)

    saved_elements = (hydrogen, carbon)

    for element in saved_elements:
        for chrg in range(element.atomic_number + 1):
            path = os.path.join(profiles_dir, "{}{:d}.json".format(element.name, chrg))

            with open(path, "r") as fhl:
                file_data = json.load(fhl)
                element_name = file_data["element"]
                charge = file_data["charge"]
                edge_data["composition"][element_name][charge] = file_data

    return edge_data.freeze()