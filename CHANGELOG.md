Project Changelog
=================

Release 1.5.0 (27 Aug 2024)
-------------------

API changes:
* The line shape models are moved to a dedicated submodule. The user code should not be affected though. (#396)
* The line shape models now have AtomicData as a required parameter.
* The method show_supported_transitions() of StarkBroadenedLine and ParametrisedZeemanTriplet is removed.
* The argument stark_model_coefficients of StarkBroadenedLine is now a tuple instead of a dict.
* The argument line_parameters of ParametrisedZeemanTriplet is now a tuple instead of a dict.

New:
* Support Raysect 0.8
* Cython version 3 is now required to build the package.
* Add custom line shape support to BeamCXLine model. (#394)
* Add PeriodicTransformXD and VectorPeriodicTransformXD functions to support the data simulated with periodic boundary conditions. (#387)
* Add CylindricalTransform and VectorCylindricalTransform to transform functions from cylindrical to Cartesian coordinates. (#387)
* Add numerical integration of Bremsstrahlung spectrum over a spectral bin. (#395)
* Replace the coarse numerical constant in the Bremsstrahlung model with an exact expression. (#409)
* Add the kind attribute to RayTransferPipelineXD that determines whether the ray transfer matrix is multiplied by sensitivity ('power') or not ('radiance'). (#412)
* Improved parsing of metadata from the ADAS ADF15 'bnd' files for H-like ions. Raises a runtime error if the metadata cannot be parsed. (#424)
* **Beam dispersion calculation has changed from sigma(z) = sigma + z * tan(alpha) to sigma(z) = sqrt(sigma^2 + (z * tan(alpha))^2) for consistancy with the Gaussian beam model. Attention!!! The results of BES and CX spectroscopy are affected by this change. (#414)**
* Improved beam direction calculation to allow for natural broadening of the BES line shape due to beam divergence. (#414)
* Add kwargs to invert_regularised_nnls to pass them to scipy.optimize.nnls. (#438)
* StarkBroadenedLine now supports Doppler broadening and Zeeman splitting. (#393)
* Add the power radiated in spectral lines due to charge exchange with thermal neutral hydrogen to the TotalRadiatedPower model. (#370)
* Add thermal charge-exchange emission model. (#57)
* PECs for C VI spectral lines for n <= 5 are now included in populate(). Rerun populate() after upgrading to 1.5 to update the atomic data repository.
* All interpolated atomic rates now return 0 if plasma parameters <= 0, which matches the behaviour of emission models. (#450)

Bug fixes:
* Fix deprecated transforms being cached in LaserMaterial after laser.transform update (#420)
* Fix IRVB calculate sensitivity method.
* Fix missing donor_metastable attribute in the core BeamCXPEC class (#411).
* **Fix the receiver ion density being passed to the BeamCXPEC instead of the total ion density in the BeamCXLine. Also fix incorrect BeamCXPEC dosctrings. Attention!!! The results of CX spectroscopy are affected by this change. (#441)**

Release 1.4.0 (3 Feb 2023)
-------------------

API changes:
* Spectroscopic observers and their groups are deprecated and replaced by groups based on Raysect's 0D observers. (#332)
* Support for Python 3.6 is dropped. It may still work, but is no longer actively tested.

Bug fixes:
* Fix and improve OpenCL utility functions. (#358)
* Fixed Bremsstrahlung trapezium evaluation (#384).

New:
* Make f_profile (current flux) a read-only attribute of EFITEquilibrium. (#355)
* Add group observer class for each of Raysect's 0D observers. (#332)
* Add a demo for observer group handling and plotting.
* Add verbose parameter to SartOpencl solver (default is False). (#358)
* Add Thomson Scattering model. (#97)
* Add Generomak core plasma profiles. (#360)
* Add toroidal_mesh_from_polygon for making mesh for not fully-360 degrees axisymmetric elements. (#365)
* Add common spectroscopic instruments: Polychromator, SurveySpectrometer, CzernyTurnerSpectrometer. (#299)
* Add new classes for free-free Gaunt factors and improve accuracy of the Gaunt factor used in Bremsstrahlung emission model. (#352)
* Add GaussianQuadrature integration method for Function1D. (#366)
* Add integrator attribute to LineShapeModel to use with lineshapes that cannot be analytically integrated over a spectral bin. (#366)
* Add a numerical integration of StarkBroadenedLine over the spectral bin. (#366)
* Add Generomak full plasma profiles obtained by blending the core and edge profiles. (#372)
* Clean up build/install dependencies. (#353)
* Test against Python 3.10 and latest released Numpy. Drop Python 3.6 and older Numpy from tests. (#391)


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
