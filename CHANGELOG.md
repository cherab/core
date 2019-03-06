Project Changelog
=================

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
