
.. _mastu_solps_plasma:

Loading a MAST-U plasma from SOLPS
==================================

Example of how to load a plasma from an existing SOLPS simulation
(`demo file <https://github.com/cherab/solps/blob/master/demos/mastu_solps_plasma.py>`_).
Also, shows how to inspect the plasma parameters. Start by importing
all required modules. ::

    import matplotlib.pyplot as plt
    import numpy as np
    from cherab.core.atomic.elements import carbon, deuterium
    from cherab.solps import SOLPSSimulation
    plt.ion()

Import the SOLPS simulation from AUG MDSplus server. ::

    mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
    ref_number = 69636

    sim = SOLPSSimulation.load_from_mdsplus(mds_server, ref_number)
    plasma = sim.plasma
    mesh = sim.mesh
    vessel = mesh.vessel

The plasma parameters are loaded automatically from the simulation. We can now access an individual species using the
`get_species` method. It takes as argument the element/isotope of interest and the desired charge state. If the species
requested was not in the simulation, an exception will be thrown. ::

    d0 = plasma.composition.get(deuterium, 0)
    d1 = plasma.composition.get(deuterium, 1)
    c0 = plasma.composition.get(carbon, 0)
    c1 = plasma.composition.get(carbon, 1)
    c2 = plasma.composition.get(carbon, 2)
    c3 = plasma.composition.get(carbon, 3)
    c4 = plasma.composition.get(carbon, 4)
    c5 = plasma.composition.get(carbon, 5)
    c6 = plasma.composition.get(carbon, 6)

The electron distribution is the only exception to the above access pattern. It is accessed with a dedicated plasma
attribute, `plasma.electron_distribution`.

Now we will sample the different species values over a regular grid in the poloidal plane. Results for density,
temperature and velocity are plotted, along with the SOLPS simulation grid. ::

    xl, xu = (0.0, 2.0)
    yl, yu = (-2.0, 2.0)
    te_samples = np.zeros((500, 500))
    ne_samples = np.zeros((500, 500))
    d0_samples = np.zeros((500, 500))
    d1_samples = np.zeros((500, 500))
    c0_samples = np.zeros((500, 500))
    c1_samples = np.zeros((500, 500))
    c2_samples = np.zeros((500, 500))
    c3_samples = np.zeros((500, 500))
    c4_samples = np.zeros((500, 500))
    c5_samples = np.zeros((500, 500))
    c6_samples = np.zeros((500, 500))
    d0_velocity = np.zeros((500, 500))
    d1_velocity = np.zeros((500, 500))
    xrange = np.linspace(xl, xu, 500)
    yrange = np.linspace(yl, yu, 500)


    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            ne_samples[j, i] = plasma.electron_distribution.density(x, 0.0, y)
            te_samples[j, i] = plasma.electron_distribution.effective_temperature(x, 0.0, y)
            d0_samples[j, i] = d0.distribution.density(x, 0.0, y)
            d1_samples[j, i] = d1.distribution.density(x, 0.0, y)
            c0_samples[j, i] = c0.distribution.density(x, 0.0, y)
            c1_samples[j, i] = c1.distribution.density(x, 0.0, y)
            c2_samples[j, i] = c2.distribution.density(x, 0.0, y)
            c3_samples[j, i] = c3.distribution.density(x, 0.0, y)
            c4_samples[j, i] = c4.distribution.density(x, 0.0, y)
            c5_samples[j, i] = c5.distribution.density(x, 0.0, y)
            c6_samples[j, i] = c6.distribution.density(x, 0.0, y)
            # magnitude of velocity vector
            d0_velocity[j, i] = d0.distribution.bulk_velocity(x, 0.0, y).length
            d1_velocity[j, i] = d1.distribution.bulk_velocity(x, 0.0, y).length

    # Turn on interactive plotting.
    plt.ion()

    mesh.plot_mesh()
    plt.title('mesh geometry')

    plt.figure()
    plt.imshow(ne_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("electron density")
    plt.figure()
    plt.imshow(te_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("electron temperature")

    plt.figure()
    plt.imshow(d0_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("D0 density")
    plt.figure()
    plt.imshow(d1_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("DI density")

    plt.figure()
    plt.imshow(c0_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CI density")
    plt.figure()
    plt.imshow(c1_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CII density")
    plt.figure()
    plt.imshow(c2_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CIII density")
    plt.figure()
    plt.imshow(c3_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CIV density")
    plt.figure()
    plt.imshow(c4_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CV density")
    plt.figure()
    plt.imshow(c5_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CVI density")
    plt.figure()
    plt.imshow(c6_samples, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("CVII density")

    plt.figure()
    plt.imshow(d0_velocity, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("D0 velocity")

    plt.figure()
    plt.imshow(d1_velocity, extent=[xl, xu, yl, yu], origin='lower')
    plt.colorbar()
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    plt.title("D1 velocity")


.. figure:: ./species_wide.png
   :align: center

   Some example plots of the plasma's temperature and density profiles.
