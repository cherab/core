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


def plot_equilibrium(equilibrium, resolution=0.025):
    """
    Generates some overview plots of a given EFIT equilibrium.

    Generates plots of normalised psi,

    :param equilibrium: The input EFIT equilibrium object.
    :param float resolution: Spatial resolution for sampling.
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

    print("Sampling B-field...")
    _, _, b = samplevector2d(eq.b_field, (rmin, rmax, nr), (zmin, zmax, nz))

    print("Sampling LCFS interior...")
    _, _, inside = sample2d(eq.inside_lcfs, (rmin, rmax, nr), (zmin, zmax, nz))

    print("Calculating B-field magnitude...")
    bx = b[:, :, 0]
    by = b[:, :, 1]
    bz = b[:, :, 2]
    bmag = np.sqrt(bx**2 + by**2 + bz**2)

    print("Plotting...")
    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(psi_sampled), cmap='jet', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('(Normalised Psi')

    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(bx), cmap='gray', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(bx), 25)
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Magnetic Field: X Component')

    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(by), cmap='gray', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(by), 25)
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Magnetic Field: Y Component')

    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(bz), cmap='gray', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(bz), 25)
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Magnetic Field: Z Component')

    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(bmag), cmap='gray', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(bmag), 25)
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Magnetic Field Magnitude')

    plt.figure()
    plt.axes(aspect='equal')
    plt.pcolormesh(r, z, np.transpose(inside), cmap='gray', shading='gouraud')
    plt.autoscale(tight=True)
    plt.colorbar()
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Inside LCFS')

    plt.figure()
    plt.axes(aspect='equal')
    plt.quiver(r[::4], z[::4], np.transpose(bx[::4, ::4]), np.transpose(bz[::4, ::4]), angles='xy', scale_units='xy', pivot='middle')
    plt.autoscale(tight=True)
    plt.contour(r, z, np.transpose(psi_sampled), levels=[1.0])
    plt.title('Poloidal Magnetic Field')

    p2r_psin, p2r_r = sample1d(eq.psin_to_r, (0, 1, 1000))

    plt.figure()
    plt.plot(p2r_psin, p2r_r)
    plt.title('Psi (Normalised) vs Outboard Major Radius')

    plt.show()
