"""
Contains functions required to perform ADMT regularisation.

This is based on work by L. C. Ingesson: see JET-R(99)08 available at
http://www.euro-fusionscipub.org/wp-content/uploads/2014/11/JETR99008.pdf
for details.
"""
import numpy as np


def generate_derivative_operators(voxel_coords, grid_index_1d_to_2d_map,
                                  grid_index_2d_to_1d_map):
    r"""
    Generate the first and second derivative operators for a regular grid.

    :param ndarray voxel_coords: an Nx2 array of coordinates of the
    centre of each voxel, (R, Z)
    :param dict grid_flat_to_2d_map: a mapping from the 1D array of
    voxels in the grid to a 2D array of voxels if they were arranged
    spatially.
    :param dict grid_2d_to_1d_map: the inverse mapping from a 2D
    spatially-arranged array of voxels to the 1D array.

    :return dict operators: a dictionary containing the derivative
    operators: Dij for i, y ∊ (x, y) and Di for i ∊ (x, y).

    This function assumes that all voxels are rectilinear, with their
    axes aligned to the coordinate axes. If this is not the case, the
    results will be nonsense.

    The return dict contains all the first and second derivative
    operators:

    .. math::
        D_{xx} \equiv \frac{\partial^2}{\partial x^2}\\
        D_{xy} \equiv \frac{\partial^2}{\partial x \partial y}

    etc.
    """
    num_cells = len(voxel_coords)
    cell_centres = np.mean(voxel_coords, axis=1)
    # Individual derivative operators
    Dx = np.zeros((num_cells, num_cells))
    Dy = np.zeros((num_cells, num_cells))
    Dxx = np.zeros((num_cells, num_cells))
    Dxy = np.zeros((num_cells, num_cells))
    Dyy = np.zeros((num_cells, num_cells))
    # TODO: for now, we assume all voxels have rectangular cross sections
    # which are approximately identical. As per Ingesson's notation, we
    # assume voxels are ordered from top left to bottom right, in column-major
    # order with each successive voxel in a column below the previous one.
    # We should try to support voxel grids of different voxel sizes too.
    dx, dy = np.diff(cell_centres[:2], axis=0).flatten()
    dx = np.min(abs(dx[dx != 0])).item()
    dy = -np.min(abs(dy[dy != 0])).item()

    # Note that iy increases as y decreases (cells go from top to bottom),
    # which is the same as Ingesson's notation in equations 37-41
    # Use the second version of the second derivative boundary formulae, so
    # that we only need to consider nearest neighbours
    for ith_cell in range(num_cells):
        at_top, at_bottom, at_left, at_right = False, False, False, False
        n_left, n_right, n_below, n_above = np.nan, np.nan, np.nan, np.nan
        n_above_left, n_above_right, n_below_left, n_below_right = np.nan, np.nan, np.nan, np.nan
        # get the 2D mesh coordinates of this cell
        ix, iy = grid_index_1d_to_2d_map[ith_cell]

        try:
            n_left = grid_index_2d_to_1d_map[ix - 1, iy]  # neighbour 1, left of n0
        except KeyError:
            at_left = True
        else:
            Dx[ith_cell, n_left] = -1 / 2
            Dxx[ith_cell, n_left] = 1

        try:
            n_below_left = grid_index_2d_to_1d_map[ix - 1, iy + 1]  # neighbour 2, below left of n0
        except KeyError:
            pass
            # KeyError does not necessarily mean bottom AND left
        else:
            Dxy[ith_cell, n_below_left] = 1 / 4

        try:
            n_below = grid_index_2d_to_1d_map[ix, iy + 1]  # neighbour 3, below n0
        except KeyError:
            at_bottom = True
        else:
            Dy[ith_cell, n_below] = -1 / 2
            Dyy[ith_cell, n_below] = 1

        try:
            n_below_right = grid_index_2d_to_1d_map[ix + 1, iy + 1]  # neighbour 4, below right of n0
        except KeyError:
            pass
        else:
            Dxy[ith_cell, n_below_right] = -1 / 4

        try:
            n_right = grid_index_2d_to_1d_map[ix + 1, iy]  # neighbour 5, right of n0
        except KeyError:
            at_right = True
        else:
            Dx[ith_cell, n_right] = 1 / 2
            Dxx[ith_cell, n_right] = 1

        try:
            n_above_right = grid_index_2d_to_1d_map[ix + 1, iy - 1]  # neighbour 6, above right of n0
        except KeyError:
            pass
        else:
            Dxy[ith_cell, n_above_right] = 1 / 4

        try:
            n_above = grid_index_2d_to_1d_map[ix, iy - 1]  # neighbour 7, above n0
        except KeyError:
            at_top = True
        else:
            Dy[ith_cell, n_above] = 1 / 2
            Dyy[ith_cell, n_above] = 1

        try:
            n_above_left = grid_index_2d_to_1d_map[ix - 1, iy - 1]  # neighbour 8, above left of n0
        except KeyError:
            pass
        else:
            Dxy[ith_cell, n_above_left] = -1 / 4

        top_left = at_top and at_left
        top_right = at_top and at_right
        bottom_left = at_bottom and at_left
        bottom_right = at_bottom and at_right

        Dxx[ith_cell, ith_cell] = -2
        Dyy[ith_cell, ith_cell] = -2

        if at_left:
            Dx[ith_cell, ith_cell] = -1
            Dx[ith_cell, n_right] = 1
            Dxx[ith_cell, ith_cell] = -1
            Dxx[ith_cell, n_right] = 1
            if not (top_left or bottom_left):
                Dxy[ith_cell, n_above_right] = 1 / 2
                Dxy[ith_cell, n_below] = 1 / 2
                Dxy[ith_cell, n_above] = -1 / 2
                Dxy[ith_cell, n_below_right] = -1 / 2

        if at_right:
            Dx[ith_cell, n_left] = -1
            Dx[ith_cell, ith_cell] = 1
            Dxx[ith_cell, n_left] = -1
            Dxx[ith_cell, ith_cell] = 1
            if not (top_right or bottom_right):
                Dxy[ith_cell, n_above] = 1 / 2
                Dxy[ith_cell, n_below_left] = 1 / 2
                Dxy[ith_cell, n_above_left] = -1 / 2
                Dxy[ith_cell, n_below] = -1 / 2

        if at_top:
            Dy[ith_cell, n_below] = -1
            Dy[ith_cell, ith_cell] = 1
            Dyy[ith_cell, n_below] = -1
            Dyy[ith_cell, ith_cell] = 1
            if not (top_left or top_right):
                Dxy[ith_cell, n_right] = 1 / 2
                Dxy[ith_cell, n_below_left] = 1 / 2
                Dxy[ith_cell, n_left] = -1 / 2
                Dxy[ith_cell, n_below_right] = -1 / 2

        if at_bottom:
            Dy[ith_cell, n_above] = 1
            Dy[ith_cell, ith_cell] = -1
            Dyy[ith_cell, n_above] = 1
            Dyy[ith_cell, ith_cell] = -1
            if not (bottom_left or bottom_right):
                Dxy[ith_cell, n_above_right] = 1 / 2
                Dxy[ith_cell, n_left] = 1 / 2
                Dxy[ith_cell, n_above_left] = -1 / 2
                Dxy[ith_cell, n_right] = -1 / 2

        if top_left:
            Dxy[ith_cell, n_below] = 1
            Dxy[ith_cell, n_right] = 1
            Dxy[ith_cell, ith_cell] = -1
            Dxy[ith_cell, n_below_right] = -1

        if top_right:
            Dxy[ith_cell, ith_cell] = 1
            Dxy[ith_cell, n_below_left] = 1
            Dxy[ith_cell, n_left] = -1
            Dxy[ith_cell, n_below] = -1

        if bottom_left:
            Dxy[ith_cell, n_above_right] = 1
            Dxy[ith_cell, ith_cell] = 1
            Dxy[ith_cell, n_above] = -1
            Dxy[ith_cell, n_right] = -1

        if bottom_right:
            Dxy[ith_cell, n_above] = 1
            Dxy[ith_cell, n_left] = 1
            Dxy[ith_cell, ith_cell] = -1
            Dxy[ith_cell, n_above_left] = -1

    Dx = Dx / dx
    Dy = Dy / dy
    Dxx = Dxx / dx**2
    Dyy = Dyy / dy**2
    Dxy = Dxy / (dx * dy)

    # Package all operators up into a dictionary
    operators = dict(Dx=Dx, Dy=Dy, Dxx=Dxx, Dyy=Dyy, Dxy=Dxy)
    return operators


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
