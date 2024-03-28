"""
Contains functions required to perform ADMT regularisation.

This is based on work by L. C. Ingesson: see JET-R(99)08 available at
http://www.euro-fusionscipub.org/wp-content/uploads/2014/11/JETR99008.pdf
for details.
"""

# Copyright 2016-2018 Euratom
# Copyright 2016-2018 United Kingdom Atomic Energy Authority
# Copyright 2016-2018 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

__author__ = "Jack Lovell, Oak Ridge National Laboratory"

from collections.abc import Mapping
import numpy as np


def generate_derivative_operators(voxel_vertices, grid_index_1d_to_2d_map,
                                  grid_index_2d_to_1d_map):
    r"""
    Generate the first and second derivative operators for a regular grid.

    :param ndarray voxel_vertices: an Nx4x2 array of coordinates of the
        vertices of each voxel, (R, Z)
    :param dict grid_1d_to_2d_map: a mapping from the 1D array of
        voxels in the grid to a 2D array of voxels if they were arranged
        spatially.
    :param dict grid_2d_to_1d_map: the inverse mapping from a 2D
        spatially-arranged array of voxels to the 1D array.

    :return: a dictionary containing the derivative operators: Dij for
        i, j ∊ (x, y) and Di for i ∊ (x, y), Dsp and Dsm.

    This function assumes that all voxels are rectilinear, with their
    axes aligned to the coordinate axes. Additionally, all voxels are
    assumed to have the same width and height. If this is not the case,
    the results will be nonsense.

    The mappings between the 1D list of voxel coordinate and their
    order in a 2D grid assume that the 2D grid would by indexed by
    (x, y), with the y coordinate varying most quickly.

    The return dict contains all the first and second derivative
    operators:

    .. math::
        D_{xx} \equiv \frac{\partial^2}{\partial x^2}\\
        D_{xy} \equiv \frac{\partial^2}{\partial x \partial y}

    etc. It also produces two additional operators, Dsp and Dsm, for
    second derivatives on the dy/dx = 1 and dy/dx = -1 diagonals
    respectively.

    Note that the standard 2D laplacian (for isotropic regularisation)
    can be trivially calculated as follows:

    .. math::
        L = (1 - \alpha) (D_{xx} + D_{yy}) + (\alpha / 2) (D_{sp} + D_{sm})

    α = 2/3 produces the operator used in Carr et. al. RSI 89, 083506 (2018).
    α = 1/3 produces the operator with optimal isotropy.
    """
    # Input argument validation: assume rectilinear voxels
    voxel_vertices = np.asarray(voxel_vertices)
    if voxel_vertices.ndim != 3 or voxel_vertices.shape[-2] != 4 or voxel_vertices.shape[-1] != 2:
        raise TypeError("voxel_vertices must be an NxMx2 array of vertices")
    if not isinstance(grid_index_1d_to_2d_map, Mapping):
        raise TypeError("grid_index_1d_to_2d_map should be dict-like")
    if not isinstance(grid_index_2d_to_1d_map, Mapping):
        raise TypeError("grid_index_2d_to_1d_map should be dict-like")

    num_cells = voxel_vertices.shape[0]
    cell_centres = np.mean(voxel_vertices, axis=1)
    # Individual derivative operators
    Dx = np.zeros((num_cells, num_cells))
    Dy = np.zeros((num_cells, num_cells))
    Dxx = np.zeros((num_cells, num_cells))
    Dxy = np.zeros((num_cells, num_cells))
    Dyy = np.zeros((num_cells, num_cells))
    Dsp = np.zeros((num_cells, num_cells))
    Dsm = np.zeros((num_cells, num_cells))
    # TODO: for now, we assume all voxels have rectangular cross sections
    # which are approximately identical. As per Ingesson's notation, we
    # assume voxels are ordered from top left to bottom right, in column-major
    # order with each successive voxel in a column below the previous one.
    # We should try to support voxel grids of different voxel sizes too.
    cell_sizes = np.diff(cell_centres, axis=0)
    dx = cell_sizes[:, 0]
    dy = cell_sizes[:, 1]
    # dx and dy are distances in Ingesson's report, so should be positive
    dx = np.min(abs(dx[dx != 0])).item()
    dy = np.min(abs(dy[dy != 0])).item()

    # Work out how the voxels are ordered: increasing/decreasing in x/y.
    xinc, yinc = np.sign(cell_centres[-1] - cell_centres[0])

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

        iright = ix + xinc
        ileft = ix - xinc
        iabove = iy + yinc
        ibelow = iy - yinc

        try:
            n_left = grid_index_2d_to_1d_map[ileft, iy]  # left of n0
        except KeyError:
            at_left = True
        else:
            Dx[ith_cell, n_left] = -1 / 2
            Dxx[ith_cell, n_left] = 1

        try:
            n_below_left = grid_index_2d_to_1d_map[ileft, ibelow]  # below left of n0
        except KeyError:
            # KeyError does not necessarily mean bottom AND left
            pass
        else:
            Dxy[ith_cell, n_below_left] = 1 / 4

        try:
            n_below = grid_index_2d_to_1d_map[ix, ibelow]
        except KeyError:
            at_bottom = True
        else:
            Dy[ith_cell, n_below] = -1 / 2
            Dyy[ith_cell, n_below] = 1

        try:
            n_below_right = grid_index_2d_to_1d_map[iright, ibelow]
        except KeyError:
            pass
        else:
            Dxy[ith_cell, n_below_right] = -1 / 4

        try:
            n_right = grid_index_2d_to_1d_map[iright, iy]
        except KeyError:
            at_right = True
        else:
            Dx[ith_cell, n_right] = 1 / 2
            Dxx[ith_cell, n_right] = 1

        try:
            n_above_right = grid_index_2d_to_1d_map[iright, iabove]
        except KeyError:
            pass
        else:
            Dxy[ith_cell, n_above_right] = 1 / 4

        try:
            n_above = grid_index_2d_to_1d_map[ix, iabove]
        except KeyError:
            at_top = True
        else:
            Dy[ith_cell, n_above] = 1 / 2
            Dyy[ith_cell, n_above] = 1

        try:
            n_above_left = grid_index_2d_to_1d_map[ileft, iabove]
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

        if np.isnan(n_above_left) and not np.isnan(n_below_right):
            Dsm[ith_cell, ith_cell] = -1
            Dsm[ith_cell, n_below_right] = 1
        elif np.isnan(n_below_right) and not np.isnan(n_above_left):
            Dsm[ith_cell, ith_cell] = -1
            Dsm[ith_cell, n_above_left] = 1
        elif np.isnan(n_above_left) and np.isnan(n_below_right):
            Dsm[ith_cell, ith_cell] = 0
        else:
            Dsm[ith_cell, ith_cell] = -2
            Dsm[ith_cell, n_above_left] = 1
            Dsm[ith_cell, n_below_right] = 1

        if np.isnan(n_above_right) and not np.isnan(n_below_left):
            Dsp[ith_cell, ith_cell] = -1
            Dsp[ith_cell, n_below_left] = 1
        elif np.isnan(n_below_left) and not np.isnan(n_above_right):
            Dsp[ith_cell, ith_cell] = -1
            Dsp[ith_cell, n_above_right] = 1
        elif np.isnan(n_below_left) and np.isnan(n_above_right):
            Dsp[ith_cell, ith_cell] = 0
        else:
            Dsp[ith_cell, ith_cell] = -2
            Dsp[ith_cell, n_above_right] = 1
            Dsp[ith_cell, n_below_left] = 1

    Dx = Dx / dx
    Dy = Dy / dy
    Dxx = Dxx / dx**2
    Dyy = Dyy / dy**2
    Dxy = Dxy / (dx * dy)
    Dsp = Dsp / (dx**2 + dy**2)
    Dsm = Dsm / (dx**2 + dy**2)

    # Package all operators up into a dictionary
    operators = dict(Dx=Dx, Dy=Dy, Dxx=Dxx, Dyy=Dyy, Dxy=Dxy, Dsp=Dsp, Dsm=Dsm)
    return operators


def calculate_admt(voxel_radii, derivative_operators, psi_at_voxels, dx, dy, anisotropy=10):
    r"""
    Calculate the ADMT regularisation operator.

    :param ndarray voxel_radii: a 1D array of the radius at the centre
        of each voxel in the grid
    :param tuple derivative_operators: a named tuple with the derivative
        operators for the grid, as returned by :func:generate_derivative_operators
    :param ndarray psi_at_voxels: the magnetic flux at the centre of
        each voxel in the grid
    :param float dx: the width of each voxel.
    :param float dy: the height of each voxel
    :param float anisotropy: the ratio of the smoothing in the parallel
        and perpendicular directions.
    :return: the ADMT regularisation operator.

    The degree of anisotropy dictates the relative suppression of
    gradients in the directions parallel and perpendicular to the
    magnetic field. For example, ``anisotropy=10`` implies parallel
    gradients in the solution are 10 times smaller than perpendicular
    gradients.

    This function assumes that all voxels are rectilinear, with their
    axes aligned to the coordinate axes. Additionally, all voxels are
    assumed to have the same width and height. If this is not the case,
    the results will be nonsense.

    N.B. the expression for the ADMT operator is taken from equation
    56 of Ingesson's report, where the ADMT operator L satisfies:

    .. math::
        \Omega = L^T \cdot L

    This means it is suitable for use in Cherab's inversion methods,
    such as NNLS and SART.
    """
    Dpar = np.full(psi_at_voxels.shape, 1)
    Dperp = Dpar / anisotropy
    Dx = derivative_operators["Dx"]
    Dy = derivative_operators["Dy"]
    Dxx = derivative_operators["Dxx"]
    Dxy = derivative_operators["Dxy"]
    Dyy = derivative_operators["Dyy"]
    normalisation = (Dx @ psi_at_voxels)**2 + (Dy @ psi_at_voxels)**2
    dpsidx = Dx @ psi_at_voxels
    dpsidy = Dy @ psi_at_voxels
    dpsidxdy = Dxy @ psi_at_voxels
    dpsidxx = Dxx @ psi_at_voxels
    dpsidyy = Dyy @ psi_at_voxels
    ddperpdx = Dx @ Dperp
    ddperpdy = Dy @ Dperp
    ddpardx = Dx @ Dpar
    ddpardy = Dy @ Dpar
    cxx = (Dperp * (dpsidx)**2 + Dpar * (dpsidy)**2) / normalisation
    cyy = (Dperp * (dpsidy)**2 + Dpar * (dpsidx)**2) / normalisation
    cxy = (Dperp - Dpar) * (dpsidx * dpsidy) / normalisation
    ddiff_term_cx = (
        dpsidx**2 * ddperpdx + dpsidy**2 * ddpardx
        + (dpsidx * dpsidy) * (ddperpdy - ddpardy)
    )
    dnorm_term_cx = -2 / normalisation * (
        (Dperp * dpsidx**2 + Dpar * dpsidy**2) * (dpsidx * dpsidxx + dpsidy * dpsidyy)
        + (Dperp - Dpar) * (dpsidx * dpsidy) * (dpsidx * dpsidxdy + dpsidy * dpsidyy)
    )
    ddiff_term_cy = (
        dpsidy**2 * ddperpdy + dpsidx**2 * ddpardy
        + (dpsidx * dpsidy) * (ddperpdx - ddpardx)
    )
    dnorm_term_cy = -2 / normalisation * (
        (Dperp * dpsidy**2 + Dpar * dpsidx**2) * (dpsidx * dpsidxdy + dpsidy * dpsidyy)
        + (Dperp - Dpar) * (dpsidx * dpsidy) * (dpsidx * dpsidxx + dpsidy * dpsidxdy)
    )
    toroidal_term_cx = cxx / voxel_radii * normalisation
    toroidal_term_cy = cxy / voxel_radii * normalisation
    cx = (
        2 * Dperp * dpsidxx * dpsidx
        + 2 * Dpar * dpsidxdy * dpsidy
        + (Dperp - Dpar) * (dpsidxdy * dpsidy + dpsidyy * dpsidx)
        + ddiff_term_cx + dnorm_term_cx + toroidal_term_cx
    ) / normalisation
    cy = (
        2 * Dperp * dpsidyy * dpsidy
        + 2 * Dpar * dpsidxdy * dpsidx
        + (Dperp - Dpar) * (dpsidxdy * dpsidx + dpsidxx * dpsidy)
        + ddiff_term_cy + dnorm_term_cy + toroidal_term_cy
    ) / normalisation
    cx = np.diag(cx)
    cy = np.diag(cy)
    cxx = np.diag(cxx)
    cyy = np.diag(cyy)
    cxy = np.diag(cxy)
    admt_operator = cx @ Dx + cy @ Dy + cxx @ Dxx + 2 * cxy @ Dxy + cyy @ Dyy
    admt_operator *= np.sqrt(dx * dy)
    return admt_operator
