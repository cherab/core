import os
import json

from raysect.core import Point2D

from cherab.tools.equilibrium import EFITEquilibrium

def load_equilibrium(file_path=None):
    """ Load Generomak EFITEquilibrium.

        :param str file_path: Path to the json equilibrium file (optional)
        :return: EFITEquilibrium
    """

    if file_path is None:
        equilibrium_folder = os.path.dirname(__file__)
        file_path = os.path.join(equilibrium_folder, "data/generomak_equilibrium.json")

    with open(file_path, "r") as fhl:
        equi_data = json.load(fhl)

    # re-create Point2D for points required by the equilibrium
    equi_data["magnetic_axis"] = Point2D(*equi_data["magnetic_axis"])
    equi_data["x_points"] = [Point2D(*equi_data["x_points"][0])]
    equi_data["strike_points"] = [Point2D(*equi_data["strike_points"][0]),
                                    Point2D(*equi_data["strike_points"][1])]

    equilibrium = EFITEquilibrium(
        r=equi_data["r"], z=equi_data["z"], psi_grid=equi_data["psi_grid"],
        psi_axis=equi_data["psi_axis"], psi_lcfs=equi_data["psi_lcfs"],
        magnetic_axis=equi_data["magnetic_axis"],
        x_points=equi_data["x_points"], strike_points=equi_data["strike_points"],
        f_profile=equi_data["f_profile"], q_profile=equi_data["q_profile"],
        b_vacuum_radius=equi_data["b_vacuum_radius"],
        b_vacuum_magnitude=equi_data["b_vacuum_magnitude"],
        lcfs_polygon=equi_data["lcfs_polygon"],
        limiter_polygon=equi_data["limiter_polygon"],
        time=equi_data["time"])

    return equilibrium
