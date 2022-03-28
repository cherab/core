Project Changelog
=================

Release 1.4.0 (TBD)
-------------------

API changes:
* Spectroscopic observers and their groups are deprecated and replaced by groups based on Raysect's 0D observers. (#332) 

New:
* Make f_profile (current flux) a read-only attribute of EFITEquilibrium. (#355)
* Add group observer class for each of Raysect's 0D observers. (#332)
* Add a demo for observer group handling and plotting.
* Add verbose parameter to SartOpencl solver (default is False). (#358)
* Add Generomak core plasma profiles. (#360)
* Add toroidal_mesh_from_polygon for making mesh for not fully-360 degrees axisymmetric elements. (#365)

Bug Fixes:
----------
* Fixed generomak plasma edge data paths.
* Fix and improve OpenCL utility functions. (#358)


Release 1.3.0 (8 Dec 2021)
--------------------------

API changes:
* Use of Cherab's interpolators is now deprecated in favour of those upstream in Raysect.

Bug fixes:
* Fixed OpenADAS repository populate() method failing with the pure python interpreter but working in ipython. (#186)
* Fix loss of precision in OpenCL SART inversion. (#188)
* Tidy up bolometer API and fix sightline orientation. (#189, #191, #193, #195)
* Fix faulty periodicity check in cylindrical ray transfer classes. (#226)
* Use correct species temperature when calculating line broadening. (#236)
* Fix corrupting JSON file when populating OpenADAS repository (#244)
* Fix getting incorrect beam population rates from OpenADAS (#253)
* Fix the Gaunt factor in the Bremsstrahlung class. (#271)
* Fix wrong intensity_s1 coefficient in BeamEmissionMultiplet. (#277)
* Fix beam emission rate calculation. (#278)
* Fix some crashes when parameters are not correctly specified. (#287, #289)
* Fix mapping from flux coordinates to Cartesian. (#302)
* Use isotope rather than element wavelengths. (#296)
* Fix some documentation typos. (#309)
* Fix misleading error messages arising from sanity checks in plasma, beam and atomic models. (#333)


New:
* Add electron rest mass and classical radius constants. (#187)
* Improve performance of ray transfer emitters and integrators (#198)
* Add some bolometry examples to the demos and documentation (#161)
* Standardise on the name "Cherab" rather than "CHERAB"
* Comaptibility with Raysect 0.7.1
* Use Raysect's function framework where possible. (#214)
* Tools to create slab plasmas. (#208)
* Use PEP517-compliant pyproject.toml to improve installation process. (#250)
* Add more accurate hydrogen isotope wavelengths. (#265)
* Add the "Generomak", an example machine with first wall mesh (#268, #312), magnetic equilibrium (#335, #341) and edge plasma profiles (#337).
* Enable specifying the value outside the LCFS when mapping to EFIT equilibria. (#270)
* Allow user to specify the path to an atomic data repository. (#291, #316)
* Switch CI from Travis to Github actions. (#280, #293)
* Add Cython definition files for Ray Transfer Emitters. (#307)
* Add spectroscopic fibre optic observers. (#284)
* Add an infra-red video bolometer (IRVB) model. (#206)
* Improve project metadata to make PyPI page more informative. (#317)
* Add Zeeman spectral line shape models. (#246)
* Use Raysect's new interpolators in place of Cherab's. (#304)
* Interpolate OpenADAS rates in log-log space. (#304)
* Use nearest neighbour extrapolation for OpenADAS ionisation and recombination rates, impact excitation and recombination PECs and line and continuum radiated power rates. (#304)





Release 1.2.0 (24 Nov 2019)
---------------------------

API changes:
* AxisymmetricVoxel vertices initialisation switched to Nx2 numpy array.
* Raysect VolumeTransforms used to handle shifts in coordinate systems rather than material specific offsets.
* Numerous minor changes (see commit history).

New:
* Merged cherab-openadas package into the core cherab package to simplify installation.
* Beam object uses a cone primitive instead of a cylinder for the bounding volume of divergent beams. 
* Added Clamp functions.
* Added ThermalCXRate.
* Added optimised ray transfer grid calculation tools.
* Added opencl optimised SART inversion to tools.
* Numerous improvements to bolometry tool chain (see commit history).

Bug fixes:
* Equilibrium normalised psi clamped to prevent negative values (occasionally caused by numerical precision issues at the core).
* trace_sightline() bug that caused repeated reintersection has been fixed.
* Numerous samller issues addressed throughout the framework (see commit history).


Release 1.1.0 (6 Mar 2019)
--------------------------

New:
* Added EFITEquilibrium class to cherab.tools.equilibrium.
* Added voxel handling utilities to cherab.tools
* Added differentials to interpolator functions.
* Added Slice2D and 3D functions to reduce the dimensions of a function object.
* Expanded list of isotopes and elements, nearly all elements/stable isotopes are now available.
* Can now look up element/isotope objects by name and/or atomic number/mass.
* Significantly expanded documentation and demos.
* Added Multiplet line-shape.

Bug fixes:
* Improved handling on non c-order arrays in various methods.
* Numerous minor bug fixes (see commit history) 


Release 1.0.1 (1 Oct 2018)
--------------------------

Bug fixes:
* Cherab package would fail if Raysect structures were altered due to using prebuilt c files. Cython is now always used to rebuild against the installed version of raysect. Cython is therefore now a dependency.


Release 1.0.0 (28 Sept 2018)
----------------------------

Initial public release.
