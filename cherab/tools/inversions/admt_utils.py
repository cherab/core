"""
Contains functions required to perform ADMT regularisation.

This is based on work by L. C. Ingesson: see JET-R(99)08 available at
http://www.euro-fusionscipub.org/wp-content/uploads/2014/11/JETR99008.pdf
for details.
"""


def generate_derivative_operators(voxel_coords, grid_1d_to_2d_map,
                                  grid_2d_to_1d_map):
    r"""
    Generate the first and second derivative operators for a regular grid.

    :param ndarray voxel_coords: an Nx2 array of coordinates of the
    centre of each voxel, (R, Z)
    :param dict grid_flat_to_2d_map: a mapping from the 1D array of
    voxels in the grid to a 2D array of voxels if they were arranged
    spatially.
    :param dict grid_2d_to_1d_map: the inverse mapping from a 2D
    spatially-arranged array of voxels to the 1D array.

    :return tuple operators: a named tuple containing the derivative
    operators: Dij for i, y ∊ (x, y) and Di for i ∊ (x, y).

    This function assumes that all voxels are rectilinear, with their
    axes aligned to the coordinate axes. If this is not the case, the
    results will be nonsense.

    The return tuple contains all the first and second derivative
    operators:

    .. math::
        D_{xx} \equiv \frac{\partial^2}{\partial x^2}\\
        D_{xy} \equiv \frac{\partial^2}{\partial x \partial y}

    etc.
    """
    return


def calculate_admt(voxel_radii, derivative_operators, psi_at_voxels, anisotropy=10):
    r"""
    Calculate the ADMT regularisation operator.

    :param ndarray voxel_radii: a 1D array of the radius at the centre
    of each voxel in the grid
    :param tuple derivative_operators: a named tuple with the derivative
    operators for the grid, as returned by :func:`generate_derivative_operators`
    :param ndarray psi_at_voxels: the magnetic flux at the centre of
    each voxel in the grid
    :param float anisotropy: the ratio of the smoothing in the parallel
    and perpendicular directions.

    :return ndarray admt: the ADMT regularisation operator.

    The degree of anisotropy dictates the relative suppression of
    gradients in the directions parallel and perpendicular to the
    magnetic field. For example, `anisotropy=10` implies parallel
    gradients in solution are 10 times smaller than perpendicular
    gradients.
    """
    return
