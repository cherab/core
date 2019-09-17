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


import numpy as np
from matplotlib import pyplot as plt

from cherab.core.math import sample1d, sample2d, samplevector2d


def _plot_summary(r, z, psi, axis, limiter, time):

    interior_contours = 10
    exterior_contours = 15

    # matplotlib uses fortran ordering
    psi = psi.transpose()

    # generate interior levels (evenly spaced between axis and boundary psi values)
    delta = 1.0 / (interior_contours + 2)
    interior_min = delta
    interior_max = 1.0 - delta
    interior_levels = np.linspace(interior_min, interior_max, interior_contours)

    # generate exterior levels
    exterior_levels = np.arange(1.0 + delta, 1.0 + (exterior_contours + 1) * delta, delta)
    fig = plt.figure()
    a = plt.axes()

    # first wall
    if limiter is not None:

        # close limiter polygon for rendering
        limiter = np.append(limiter, [limiter[0, :]], axis=0)

        # draw boundary
        a.add_artist(plt.Rectangle((r.min(), z.min()), r.max() - r.min(), z.max() - z.min(), facecolor="#e0e0e0"))
        a.add_artist(plt.Polygon(limiter, facecolor="#ffffff"))

    # contours
    plt.contour(r, z, psi, levels=interior_levels, colors="#a0a0ff", linestyles="solid", linewidths=1.2)
    plt.contour(r, z, psi, levels=[1.0], colors="#ff0000", linestyles="solid", linewidths=1.4)
    plt.contour(r, z, psi, levels=exterior_levels, colors="#c0c0c0", linestyles="solid", linewidths=0.8)
    if limiter is not None:
        plt.plot(limiter[:, 0], limiter[:, 1], color="#000000")

    # magnetic axis
    plt.plot(axis.x, axis.y, "x", color="#ff0000")
    plt.annotate("({:.3f}, {:.3f})".format(*axis), xy=(axis.x, axis.y), xytext=(0.0, -15.0),
                 textcoords="offset points", size="x-small", color="#ff0000", horizontalalignment="center",
                 bbox=dict(boxstyle="round", alpha=0.6, facecolor="#ffffff", edgecolor="none"))

    # axis labels and configuration
    plt.title('Equilibrium at time: {:.3f}s'.format(time))
    plt.xlabel("R (meters)")
    plt.ylabel("Z (meters)")
    plt.minorticks_on()
    fig.axes[0].set_aspect("equal")
    fig.axes[0].set_axisbelow(True)


def plot_equilibrium(equilibrium, detail=False, resolution=0.025):
    """
    Generates some overview plots of a given EFIT equilibrium.

    :param equilibrium: The input EFIT equilibrium object.
    :param detail: If true, prints additional information about the equilibrium.
    :param float resolution: Spatial resolution for sampling (default=0.025).

    .. code-block:: pycon

       >>> from cherab.tools.equilibrium import example_equilibrium, plot_equilibrium
       >>>
       >>> equilibrium = example_equilibrium()
       >>> plot_equilibrium(equilibrium, detail=False, resolution=0.001)
    """

    eq = equilibrium

    # plot equilibrium
    rmin, rmax = eq.r_range
    zmin, zmax = eq.z_range

    # sample every 1 cm
    nr = round((rmax - rmin) / resolution)
    nz = round((zmax - zmin) / resolution)

    print("Sampling psi...")
    r, z, psi_sampled = sample2d(eq.psi_normalised, (rmin, rmax, nr), (zmin, zmax, nz))

    print("Plotting summary...")
    _plot_summary(r, z, psi_sampled, eq.magnetic_axis, eq.limiter_polygon, eq.time)

    if detail:

        print("Sampling B-field...")
        _, _, b = samplevector2d(eq.b_field, (rmin, rmax, nr), (zmin, zmax, nz))

        print("Sampling LCFS interior...")
        _, _, inside_lcfs = sample2d(eq.inside_lcfs, (rmin, rmax, nr), (zmin, zmax, nz))

        if eq.inside_limiter:
            print("Sampling Limiter interior...")
            _, _, inside_limiter = sample2d(eq.inside_limiter, (rmin, rmax, nr), (zmin, zmax, nz))

        print("Calculating B-field magnitude...")
        bx = b[:, :, 0]
        by = b[:, :, 1]
        bz = b[:, :, 2]
        bmag = np.sqrt(bx**2 + by**2 + bz**2)

        print("Sampling q...")
        psin, q = sample1d(eq.q, (0, 1, 100))

        print("Plotting details...")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(psi_sampled), shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Normalised Psi')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(bx), cmap='gray', shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(bx), 25)
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Magnetic Field: X Component')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(by), cmap='gray', shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(by), 25)
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Magnetic Field: Y Component')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(bz), cmap='gray', shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(bz), 25)
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Magnetic Field: Z Component')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(bmag), cmap='gray', shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(bmag), 25)
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Magnetic Field Magnitude')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.pcolormesh(r, z, np.transpose(inside_lcfs), cmap='gray', shading='gouraud')
        plt.autoscale(tight=True)
        plt.colorbar()
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Inside LCFS')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        if eq.inside_limiter:
            plt.figure()
            plt.axes(aspect='equal')
            plt.pcolormesh(r, z, np.transpose(inside_limiter), cmap='gray', shading='gouraud')
            plt.autoscale(tight=True)
            plt.colorbar()
            plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
            plt.title('Inside Limiter')
            plt.xlabel("R (meters)")
            plt.ylabel("Z (meters)")

        plt.figure()
        plt.axes(aspect='equal')
        plt.quiver(r[::4], z[::4], np.transpose(bx[::4, ::4]), np.transpose(bz[::4, ::4]), angles='xy', scale_units='xy', pivot='middle')
        plt.autoscale(tight=True)
        plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
        plt.title('Poloidal Magnetic Field')
        plt.xlabel("R (meters)")
        plt.ylabel("Z (meters)")

        if eq.psin_to_r is not None:  # Only if psin is monotonic
            p2r_psin, p2r_r = sample1d(eq.psin_to_r, (0, 1, 1000))

            plt.figure()
            plt.plot(p2r_psin, p2r_r)
            plt.title('Psi Normalised vs Outboard Major Radius')
            plt.xlabel("a.u.")
            plt.ylabel("R (meters)")

        plt.figure()
        plt.plot(psin, q)
        plt.title('Safety Factor (q) vs Psi Normalised')
        plt.xlabel("Psi Normalised")
        plt.ylabel("a.u.")

    plt.show()
