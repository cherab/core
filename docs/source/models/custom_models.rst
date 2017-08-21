
Custom emission models
----------------------

Custom emitters are implemented as Raysect materials with the `VolumeEmitterInhomogeneous` class.
You model should inherit from this base class, only two methods need to be implemented,
the `__init__()` and `emission_function()` methods.

The `__init__()` method should be used to setup any plasma parameters you will need access
to for calculations. You could pass in the whole plasma object, or alternatively just the
species you need. In this example, I will need the electron distribution and atomic deuterium
distribution. Both of these can be passed in from the plasma object.

I'm also passing in a spectroscopic `Line` object which holds useful information about my
emission line. But this can change alot from application to application. We are also loading
some atomic data from ADAS. You could optionally pass in the atomic data you want to use in
the `__init__()`.

The `step` parameter is the only parameter required by the parent `VolumeEmitterInhomogeneous`
class. This parameter determines the integration step size for sampling and will need to be
adjusted based on your application's scale lengths. Make ure you call the parent init whith the
super method, e.g. `super().__init__(step=step)`.

All the magic happens in the `emission_function()` method. This method is called at every point
in space where the ray-tracer would like to know the emission. The arguments are fixed and are
as follows:

* `point` (`Point3D`) - current position in local primitive coordinates
* `direction` (`Vector3D`) - current ray direction in local primitive coordinates
* `spectrum` (`Spectrum`) - measured spectrum so far. Don't overwrite it. Add your local
  emission to the measured spectrum. Units are in spectral radiance (W/m3/str/nm).
* `world` (`World`) - the world being ray-traced. You may have multiple worlds.
* `ray` (`Ray`) - the current ray being traced.
* `primitive` (`Primitive`) - the primitive container for this material. Could be a sphere,
  cyliner, or CAD mesh for example.
* `to_local` (`AffineMatrix3D`) - Affine matrix for coordinate transformations to local coordinates.
* `to_world` (`AffineMatrix3D`) - Affine matrix for coordinate transformations to world coordinates.

.. WARNING::
   Don't override the spectrum parameter, you will loose all the previous ray samples. Instead you
   should add your local emission at the current point in space to the measured spectrum array.

Here is an example class implementation of an excitation line. ::

    class ExcitationLine(VolumeEmitterInhomogeneous):

        def __init__(self, line, electron_distribution, atom_species, step=0.005,
                     block=0, filename=None):

            super().__init__(step=step)

            self.line = line
            self.electron_distribution = electron_distribution
            self.atom_species = atom_species

            # Function to load ADAS rate coefficients
            # Replace this with your own atomic data as necessary.
            self.pec_excitation = get_adf15_pec(line, 'EXCIT', filename=filename, block=block)

        def emission_function(self, point, direction, spectrum, world, ray, primitive,
                              to_local, to_world):

            ##########################################
            # Load all data you need for calculation #

            # Get the current position in world coordinates,
            # 'point' is in local primitive coordinates by default
            x, y, z = point.transform(to_world)

            # electron density n_e(x, y, z) at current point
            ne = self.electron_distribution.density(x, y, z)
            # electron temperature t_e(x, y, z) at current point
            te = self.electron_distribution.effective_temperature(x, y, z)
            # density of neutral atoms of species specified by line.element
            na = self.atom_species.distribution.density(x, y, z)

            # Electron temperature and density must be in valid range for ADAS data.
            if not 5E13 < ne < 2E21:
                return spectrum
            if not 0.2 < te < 10000:
                return spectrum

            # Photo Emission Coefficient (PEC) for excitation at this temperature and density
            pece = self.pec_excitation(ne, te)

            # calculate line intensity
            inty = 1E6 * (pece * ne * na)

            weight = self.line.element.atomic_weight
            rest_wavelength = self.line.wavelength

            ###############################
            # Calculate the emission line #

            # Calculate a simple gaussian line at each line wavelength in spectrum
            # Add it to the existing spectrum. Don't override previous results!

            sigma = sqrt(te * ELEMENTARY_CHARGE / (weight * AMU)) * rest_wavelength / SPEED_OF_LIGHT
            i0 = inty/(sigma * sqrt(2 * PI))
            width = 2*sigma**2
            for i, wvl in enumerate(spectrum.wavelengths):
                spectrum.samples[i] += i0 * exp(-(wvl - rest_wavelength)**2 / width)

            return spectrum
