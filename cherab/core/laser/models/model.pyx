from raysect.optical cimport SpectralFunction, Spectrum, InterpolatedSF, Point3D, Vector3D
from raysect.core.math.cython.utility cimport find_index

from cherab.core.laser.node cimport Laser
from cherab.core.laser.models.model_base cimport LaserModel
from cherab.core cimport Plasma


from libc.math cimport M_PI, sqrt, exp

cdef class UniformPowerDensity(LaserModel):

    def __init__(self, power_density=1, Vector3D polarization = Vector3D(0, 1, 0), central_wavelength = 1060, spectral_sigma = 0.01, wlen_min = 1059.8,
                 wlen_max=1060.2, nbins=100):

        super().__init__()

        self._power_density = power_density
        self._central_wavelength = central_wavelength
        self._spectral_sigma = spectral_sigma
        self._wlen_min = wlen_min
        self._wlen_max = wlen_max
        self._nbins = nbins
        self._polarization_vector = Vector3D(0, 1, 0)

        self._gaussian_spectrum()


    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef double get_power_density(self, double x, double y, double z, double wavelength):
        """
        Returns the volumetric power density of the laser light at the specified point.
        The return value is a sum for all laser wavelengths.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._power_density

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector


    cpdef double get_laser_spectrum(self):
        """
        Retruns ratio of the power enclosed in the specified wavelength range
        
        :param minimum_wavelength: Lower boundary of the wavelength band. 
        :param maximum_wavelength: Upper boundary of the wavelength band. 
        :return: 
        """

        return self._laser_spectrum

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D vector):

        if vector.length < 0:
            raise ValueError("Vector of 0 length is not allowed.")

        self._polarization_vector = vector.normalise()

    def _gaussian_spectrum(self):


        self._spectrum = Spectrum(self._wlen_min, self._wlen_max, self._nbins)

        spectrum_delta = self._spectrum.delta_wavelength
#        for index, wlen in enumerate(self._spectrum.wavelengths):
#            self._spectrum.samples_mv[index] = np.exp(-1/2 * power(wlen - self._central_wavelength, 2)/self._spectral_sigma**2) / spectrum.delta_wavelength

cdef class GaussianBeamAxisymmetric(LaserModel):

    def __init__(self, Laser laser = None, Vector3D polarization = Vector3D(0, 1, 0), power=1,
                  laser_sigma = 0.01, waist_radius=0.001, m2 = 1, focus_z = 0.0):

        super().__init__()

        self._set_defaults()
        #laser sigma dependent constants
        #set laser constants
        self.laser_power = power
        self.polarization = polarization.normalise()
        self.waist_radius = waist_radius
        self.m2 = m2
        self.focus_z = focus_z

    def _set_defaults(self):
        self._waist_radius = 0.1
        self._waist2 = 0.001
        self._m2 = 1

    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef double get_power_density(self, double x, double y, double z, double wavelength):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        cdef:
            double power, width, r, power_axis

        r2 = x ** 2 + y ** 2

        width2 = self.get_beam_width2(z, wavelength)
        power_axis = self._power_const / width2
        power = power_axis * exp(-2 * r2 / width2)
        return power


    cpdef Spectrum get_power_density_spectrum(self, double x, double y, double z):


        cdef:
            double r2, volumetric_density, spectral_density, beam_width2
            int index
            Spectrum spectrum

        r2 = x ** 2 + y ** 2


        spectrum = self._laser_spectrum.copy()

        for index in range(spectrum.bins):
            beam_width2 = self.get_beam_width2(z, self._laser_spectrum.wavelengths[index])
            volumetric_density = self._power_const / beam_width2 *\
                                 exp(-2 * r2 / beam_width2)
            spectrum.samples_mv[index] *= volumetric_density

        return spectrum


    cpdef double get_power_axis(self, double z, double wavelength):
        return self._power_const / self.get_beam_width(z, wavelength) ** 2

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        return self._polarization_vector

    cpdef double get_beam_width2(self, double z, double wavelength):
        return self._waist2 + wavelength ** 2 * self._waist_const * (z - self._focus_z)  ** 2

    @property
    def m2 (self):
        return self._m2
    @m2.setter
    def m2(self, value):

        if not value > 0:
            raise ValueError("Laser power has to be larger than 0.")

        self._m2 = value
        self._change_waist_const()

    @property
    def laser_power(self):
        return self._laser_power

    @laser_power.setter
    def laser_power(self, value):
        if not value > 0:
            raise ValueError("Laser power has to be larger than 0.")
        self._laser_power = value
        self._power_const = 2 * value / M_PI

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D value):

        if value.length == 0:
            raise ValueError("Polarisation can't be a zero length vector.")

        self._polarization_vector = value.normalise()

    @property
    def waist_radius(self):
        return self._waist_radius

    @waist_radius.setter
    def waist_radius(self,double value):

        if not value > 0:
            raise ValueError("Value has to be larger than 0.")

        self._waist_radius = value
        self._waist2 = value ** 2
        self._change_waist_const()


    def _change_waist_const(self):
        self._waist_const = (1e-9 * self._m2 / (M_PI * self._waist_radius)) ** 2


    @property
    def laser_spectrum(self):
        return self._laser_spectrum

    @laser_spectrum.setter
    def laser_spectrum(self, Spectrum value):
        self._laser_spectrum = value

    @property
    def focus_z(self):
        return self._focus_z

    @focus_z.setter
    def focus_z(self, double value):
        self._focus_z = value

cdef class GaussianAxisymmetricalConstant(LaserModel):

    def __init__(self, Laser laser=None, Plasma plasma=None, power=0, central_wavelength = 1060,
                 spectral_sigma = 0.01, spectrum_wlen_min = 1059.8, spectrum_wlen_max=1060.2, spectrum_nbins=100,
                 laser_sigma = 0.01, Vector3D polarisation_vector = Vector3D(0, 1, 0)):

        super().__init__()

        #laser sigma dependent constants
        self._const_width = 0.1
        self._recip_laser_sigma2 = 1
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max

        #set laser constants
        self._laser_power = power
        self._spectral_mu = central_wavelength
        self._spectral_sigma = spectral_sigma
        self.spectrum_min_wavelength = spectrum_wlen_min
        self.spectrum_max_wavelength = spectrum_wlen_max
        self.spectrum_nbins = spectrum_nbins
        self.laser_sigma = laser_sigma
        self._polarization_vector = polarisation_vector.normalise()
        self._create_laser_spectrum()

    cpdef Vector3D get_pointing(self, double x, double y, double z):
        """
        Returns the pointing vector of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: Intensity in m^-3. 
        """

        return Vector3D(0, 0, 1)

    cpdef double get_power_density(self, double x, double y, double z, double wavelength):
        """
        Returns the power density of the light at the specified point.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """

        cdef:
            double r2, volumetric_density, spectral_density
            int wlen_index

        r2 = x ** 2 + y ** 2

        volumetric_density = self._const_width * exp(-0.5 * r2 * self._recip_laser_sigma2)

        wlen_index = find_index(self._laser_spectrum, wavelength)
        spectral_density = self._laser_spectrum.samples_mv[wlen_index]

        return self._laser_power * volumetric_density * spectral_density


    cpdef Spectrum get_power_density_spectrum(self, double x, double y, double z):


        cdef:
            double r2, volumetric_density, spectral_density
            int index
            Spectrum spectrum

        r2 = x ** 2 + y ** 2

        volumetric_density = self._const_width * exp(-0.5 * r2 * self._recip_laser_sigma2)

        spectrum = self._laser_spectrum.copy()

        for index in range(spectrum.nbin):
            spectrum.samples_mv[index] *= volumetric_density

        return spectrum

    cpdef Vector3D get_polarization(self, double x, double y, double z):
        """
        Returns vector denoting the laser polarisation.
        
        The point is specified in the laser beam space.
        
        :param x: x coordinate in meters.
        :param y: y coordinate in meters. 
        :param z: z coordinate in meters.
        :return: power density in Wm^-3. 
        """
        return self._polarization_vector

    @property
    def polarization(self):
        return self._polarization_vector

    @polarization.setter
    def polarization(self, Vector3D vector):

        if vector.length < 0:
            raise ValueError("Vector of 0 length is not allowed.")

        self._polarization_vector = vector.normalise()

    @property
    def spectral_sigma(self):
        return self._spectral_sigma

    @spectral_sigma.setter
    def spectral_sigma(self, value):
        if value <= 0:
            raise ValueError("Spectral sigma has to be larger than 0.")

        self._spectral_sigma = value

        self._create_laser_spectrum()

    @property
    def spectral_mu(self):
        return self._spectral_mu

    @spectral_mu.setter
    def spectral_mu(self, value):
        if value <=0:
            raise ValueError("Central wavelength of the laser has to be larger than 0.")

        self._spectral_mu = value

        self._create_laser_spectrum()

    @property
    def spectrum_nbins(self):
        return self._spectrum_nbins

    @spectrum_nbins.setter
    def spectrum_nbins(self, value):
        #verify correct value of value
        if not isinstance(value, int):
            try:
                value = int(value)
                print("Converting number of bins to integer.")
            except TypeError:
                raise TypeError("Value has to be convertable to integer.")

            if value <= 0:
                raise ValueError("Number of bins has to be larger than 0.")

        self._spectrum_nbins = value
        self._create_laser_spectrum()

    @property
    def spectrum_min_wavelength(self):
        return self._spectrum_min_wavelength


    @spectrum_min_wavelength.setter
    def spectrum_min_wavelength(self, value):

        if value >= self._spectrum_max_wavelength:
            raise ValueError("Minimum spectrum wavelength should be smaller than maximum laser-spectrum wavelength.")

        self._spectrum_min_wavelength = value

    @property
    def spectrum_max_wavelength(self):
        return self._spectrum_max_wavelength


    @spectrum_max_wavelength.setter
    def spectrum_max_wavelength(self, value):

        if value <= self._spectrum_min_wavelength:
            raise ValueError("Maximum spectrum wavelength should be larger than minimum laser-spectrum wavelength.")

        self._spectrum_max_wavelength = value

    def _create_laser_spectrum(self):

        self._laser_spectrum = Spectrum(self._spectrum_min_wavelength, self._spectrum_max_wavelength, self._spectrum_nbins)
        wavelengths = self._laser_spectrum.wavelengths
        #samples = np.exp(-1/2 * np.power(wavelengths - self._spectral_mu, 2)/self._spectral_sigma**2) / self._laser_spectrum.delta_wavelength

        #interpolated = InterpolatedSF(wavelengths, samples, normalise=True)
        #for index, value in enumerate(wavelengths):
        #    self._laser_spectrum.samples_mv[index] = interpolated(value)

        self._laser_spectrum = self._laser_spectrum


    @property
    def laser_sigma(self):
        return self._laser_sigma

    @laser_sigma.setter
    def laser_sigma(self, value):
        if value <= 0:
            raise ValueError("Standard deviation of the laser power has to be larger than 0.")

        self._laser_sigma = value

        self._const_width = 1 / (self._laser_sigma * sqrt(2 * M_PI))
        self._recip_laser_sigma2 = 1 / self._laser_sigma ** 2

    @property
    def laser_power(self):
        return self._laser_power

    @laser_power.setter
    def laser_power(self, value):
        if value <=0:
            raise ValueError("Laser power has to be larger than 0.")

        self._laser_power = value

