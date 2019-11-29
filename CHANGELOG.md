Project Changelog
=================

Release 1.2.1 (TBD)
-------------------

Bug fixes:
* Fixed OpenADAS repository populate() method failing with the pure python interpreter but working in ipython. 


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
