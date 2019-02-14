
Plasma sources
==============

The core plasma object needs to be populated with distributions for each species in the model. The two most common ways
of providing a plasma model are by using existing measured core profiles, and loading from a simulation such as SOLPS.

Measured profiles
-----------------

In this example, we use the PPF system to load a JET plasma from analysed diagnostic PPFs. Start by loading the
equilibrium and creating a psi mapping function. ::

    # Load equilibrium object
    src = DataSource()
    src.n_pulse = SHOT
    src.time = ANALYSIS_TIME
    psi = src.get_psi_normalised(exterior_value=1, cached2d=True)
    inside = lambda x, y, z: psi(x, y, z) != 1.

Use the JETMeasuredProfile class to load required input data from PPF system. ::

    # Load all required reference data for seeding plasma profiles

    # HRTS profiles
    hrts_te = JETMeasuredProfile("HRTS_te", SHOT, "HRTS", "te", psi, errvar_code="dte",
                                 filtered_profile=True, start_time=ANALYSIS_TIME)
    hrts_ne = JETMeasuredProfile("HRTS_ne", SHOT, "HRTS", "ne", psi, errvar_code="dne",
                                 filtered_profile=True, start_time=ANALYSIS_TIME)

    # KS5 profile for toroidal rotation
    vel_profile = JETMeasuredProfile("V_tor", SHOT, "CXFM", "ANGF", psi,
                                     filtered_profile=True, start_time=ANALYSIS_TIME)

Setup 3d flux functions for each species and profile type. ::

    flow_velocity_tor = IsoMapper3D(psi, vel_profile)
    ion_temperature = IsoMapper3D(psi, hrts_te)
    electron_density = IsoMapper3D(psi, hrts_ne)
    density_c6 = IsoMapper3D(psi, hrts_ne * 0.03)
    density_d = electron_density - 6 * density_c6

    flow_velocity = lambda x, y, z: Vector3D(y * flow_velocity_tor(x, y, z),
                                             - x * flow_velocity_tor(x, y, z),
                                             0.) / np.sqrt(x*x + y*y)

Load the plasma profiles into distribution functions for each species. ::

    d_distribution = Maxwellian(density_d, ion_temperature, flow_velocity,
                                deuterium.atomic_weight * atomic_mass)
    c6_distribution = Maxwellian(density_c6, ion_temperature, flow_velocity,
                                 carbon.atomic_weight * atomic_mass)
    e_distribution = Maxwellian(electron_density, ion_temperature, flow_velocity, electron_mass)

    d_species = Species(deuterium, 1, d_distribution)
    c6_species = Species(carbon, 6, c6_distribution)

Attach all the species to the plasma. ::

    plasma.inside = inside
    plasma.electron_distribution = e_distribution
    plasma.set_species([d_species, c6_species])

SOLPS Simulations
-----------------

Using an existing simulation can make it much easier to load a plasma. In this case, a reference simulation will be
loaded from the AUG MDSplus server. ::

    from cherab_contrib.simulation_data.solps.solps_plasma import SOLPSSimulation

    # Load plasma from SOLPS model
    sim = SOLPSSimulation.load_from_mdsplus(mds_server='solps-mdsplus.aug.ipp.mpg.de:8001',
                                            ref_number=40195)
    plasma = sim.plasma
    mesh = sim.mesh

If you have the raw SOLPS output, it is also possible to load the simulation from files. ::

    SIM_PATH = '/home/mcarr/mst1/aug_2016/'
    sim = SOLPSSimulation.load_from_output_files(SIM_PATH+'b2fgmtry', SIM_PATH+'b2fstate')
    mesh = sim.mesh
    plasma = sim.plasma


SOLPSMesh Class
^^^^^^^^^^^^^^^

Documentation coming soon.

.. Commented out
   autoclass:: cherab_contrib.simulation_data.SOLPSMesh
   :members:
   autoclass:: cherab_contrib.simulation_data.SOLPSSimulation
   :members:
