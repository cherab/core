import os

from raysect.core import translate, rotate_z

from raysect.primitive import import_obj

from raysect.optical.library import RoughTungsten


FIRST_WALL_COMPONENT = {}
FIRST_WALL_COMPONENT["InnerWallLimiter"] = {"component_name": "InnerWallLimiter",
                                            "file_name": "InnerWallLimiter.obj",
                                            "initial_toroidal_shift": -90,
                                            "toroidal_step": 11.25,
                                            "toroidal_instances": 32,
                                            "initial_vertical_shift": -0.561,
                                            "vertical_step": 187e-3,
                                            "vertical_instances": 10}

FIRST_WALL_COMPONENT["InnerDivertorBaffle"] = {"component_name": "InnerDivertorBaffle",
                                               "file_name": "InnerDivertorBaffle.obj",
                                               "initial_toroidal_shift": -90,
                                               "toroidal_step": 11.25,
                                               "toroidal_instances": 32,
                                               "initial_vertical_shift": 0,
                                               "vertical_step": 0,
                                               "vertical_instances": 1}

FIRST_WALL_COMPONENT["BottomBaffle"] = {"component_name": "BottomBaffle",
                                        "file_name": "BottomBaffle.obj",
                                        "initial_toroidal_shift": -90,
                                        "toroidal_step": 11.25,
                                        "toroidal_instances": 32,
                                        "initial_vertical_shift": 0,
                                        "vertical_step": 0,
                                        "vertical_instances": 1}

FIRST_WALL_COMPONENT["BottomDivertorFloor"] = {"component_name": "BottomDivertorFloor",
                                               "file_name": "BottomDivertorFloor.obj",
                                               "initial_toroidal_shift": -90,
                                               "toroidal_step": 11.25,
                                               "toroidal_instances": 32,
                                               "initial_vertical_shift": 0,
                                               "vertical_step": 0,
                                               "vertical_instances": 1}

FIRST_WALL_COMPONENT["BottomOuterVerticalTarget"] = {"component_name": "BottomOuterVerticalTarget",
                                                     "file_name": "BottomOuterVerticalTarget.obj",
                                                     "initial_toroidal_shift": -90,
                                                     "toroidal_step": 11.25,
                                                     "toroidal_instances": 32,
                                                     "initial_vertical_shift": 0,
                                                     "vertical_step": 0,
                                                     "vertical_instances": 1}

FIRST_WALL_COMPONENT["BottomInnerVerticalTarget"] = {"component_name": "BottomInnerVerticalTarget",
                                                     "file_name": "BottomInnerVerticalTarget.obj",
                                                     "initial_toroidal_shift": -90,
                                                     "toroidal_step": 11.25,
                                                     "toroidal_instances": 32,
                                                     "initial_vertical_shift": 0,
                                                     "vertical_step": 0,
                                                     "vertical_instances": 1}

FIRST_WALL_COMPONENT["TopBaffle"] = {"component_name": "TopBaffle",
                                     "file_name": "TopBaffle.obj",
                                     "initial_toroidal_shift": -90,
                                     "toroidal_step": 11.25,
                                     "toroidal_instances": 32,
                                     "initial_vertical_shift": 0,
                                     "vertical_step": 0,
                                     "vertical_instances": 1}

FIRST_WALL_COMPONENT["TopDivertorFloor"] = {"component_name": "TopDivertorFloor",
                                            "file_name": "TopDivertorFloor.obj",
                                            "initial_toroidal_shift": -90,
                                            "toroidal_step": 11.25,
                                            "toroidal_instances": 32,
                                            "initial_vertical_shift": 0,
                                            "vertical_step": 0,
                                            "vertical_instances": 1}

FIRST_WALL_COMPONENT["TopInnerVerticalTarget"] = {"component_name": "TopInnerVerticalTarget",
                                                  "file_name": "TopInnerVerticalTarget.obj",
                                                  "initial_toroidal_shift": -90,
                                                  "toroidal_step": 11.25,
                                                  "toroidal_instances": 32,
                                                  "initial_vertical_shift": 0,
                                                  "vertical_step": 0,
                                                  "vertical_instances": 1}

FIRST_WALL_COMPONENT["TopOuterVerticalTarget"] = {"component_name": "TopOuterVerticalTarget",
                                                  "file_name": "TopOuterVerticalTarget.obj",
                                                  "initial_toroidal_shift": -90,
                                                  "toroidal_step": 11.25,
                                                  "toroidal_instances": 32,
                                                  "initial_vertical_shift": 0,
                                                  "vertical_step": 0,
                                                  "vertical_instances": 1}

FIRST_WALL_COMPONENT["OuterWallLimiter"] = {"component_name": "OuterWallLimiter",
                                            "file_name": "OuterWallLimiter.obj",
                                            "initial_toroidal_shift": -45,
                                            "toroidal_step": 45,
                                            "toroidal_instances": 8,
                                            "initial_vertical_shift": 0,
                                            "vertical_step": 0,
                                            "vertical_instances": 1}


def load_component_group(file_path, parent, material, component_name,
                         toroidal_step=0, toroidal_instances=1, initial_toroidal_shift=0,
                         vertical_step=0, vertical_instances=1, initial_vertical_shift=0):
    """Adds a group of first wall componenets. The group consists of identical components which are toroidally
       and vertically distributed. The components are instances of the mesh loaded from the given
       Wavefront OBJ mesh file (.obj) and have the same material. The distribution is on a 2 dimensional matrix
       where the first dimension is the toroidal and the second is the vertical direction. The toroidal angle
       is calculated from the x axis and the component 0 is on the x axis. The vertical direction is calculated
       from the midplane and the component 0 is the bottom most component.

       :param str file_path: Path to the Wavefront OBJ mesh file (.obj) containig the first wall component mesh.
       :param Node parent: The parent node in the Raysect scene-graph.
       :param Material material: Instance of Raysect optical material given to the components within the group.
       :param str component_name: Name of the component.
       :param float toroidal_step: Toroidal angle by which the components in the group are rotated. Defaults to 0.
       :param integer toroidal_instances: Number of instances in the toroidal direction. Defaults to 1.
       :param float initial_toroidal_shift: Angle by which the whole group should be rotated in the toroidal direction.
                                            Defaults to 0.
       :param float vertical_step: Vertical distance by which the components in the group are shifted. Defaults to 0.
       :param int vertical_instances: Number of components in the verical direction. Defaults to (0, 1).
       :param float initial_vertical_shift: Distance by which the whole group is translated in the z direction.
                                            Defaults to 0.

        :return: Dictionary of the components in the group.

    """

    original = import_obj(file_path)

    component_group = {}

    for tor in range(toroidal_instances):
        for vert in range(vertical_instances):
            transform = (translate(0, 0, initial_vertical_shift + vert * vertical_step) *
                         rotate_z(initial_toroidal_shift + tor * toroidal_step))
            component_id = "{:d}, {:d}".format(tor, vert)
            name = component_name + ": " + component_id

            component_group[component_id] = original.instance(material=material, transform=transform,
                                                              parent=parent, name=name)

    return component_group


def load_first_wall(parent=None, material=RoughTungsten(0.1), mesh_folder=None):
    """ Load Generomak first wall components.

        :parameter Node parent: The parent node in the Raysect scene-graph.
        :param Material material: Instance of Raysect optical material given to the components within the group.
        :param str mesh_folder: Path to the folder containing the first wall components.

        :return: Dictionary of the Generomak first wall component groups.
    """

    if mesh_folder is None:
        generomak_folder = os.path.dirname(__file__)
        mesh_folder = os.path.join(generomak_folder, "data", "first_wall")

    components = {}

    for _, description in FIRST_WALL_COMPONENT.items():
        file_path = os.path.join(mesh_folder, description["file_name"])
        name = description["component_name"]
        components[name] = load_component_group(file_path, parent, material,
                                                name,
                                                description["toroidal_step"],
                                                description["toroidal_instances"],
                                                description["initial_toroidal_shift"],
                                                description["vertical_step"],
                                                description["vertical_instances"],
                                                description["initial_vertical_shift"]
                                                )

    return components
