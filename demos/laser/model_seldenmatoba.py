import matplotlib.pyplot as plt

from raysect.optical.spectrum import Spectrum

from cherab.core.model.laser import SeldenMatobaThomsonSpectrum
from cherab.core.utility import RecursiveDict


density = 3e19  # electron density
temperatures = [100, 2.e3]  # electron temperatures in eV
laser_wavelength = 1060  # wavelength of the laser light in nm
laser_energy = 1  # energy density of the laser light in J/m^3

# angle between observation direction and electric field
angles_pol = [90, 45]

# angles between propagation direction and observation direction
angles_obs = [45, 90, 135, 120] 

# define spectrum of observed scattering
min_wavelength = 650
max_wavelength = 1300
bins = 1000


# calculate Thomson scattered spectra for the specified combinations of el. properties and observation
model = SeldenMatobaThomsonSpectrum()
scattered = RecursiveDict()

for te in temperatures:
    for ap in angles_pol:
        for ao in angles_obs:
            spectrum = Spectrum(min_wavelength, max_wavelength, bins)
            scattered[te][ap][ao] = model.calculate_spectrum(density, te, laser_energy, laser_wavelength,
                                                             ao, ap, spectrum).samples
scattered = scattered.freeze()

wvls = spectrum.wavelengths

# plot temperature influence on scattered spectra
ap, ao = 90, 90
_, ax = plt.subplots()
ax.set_title("Scattered spectrum, pol. angl.={:d}, obs. angl = {:d}".format(ap, ao))
for te in temperatures:
    rad = scattered[te][ap][ao]
    ax.plot(wvls, rad, label="Te = {:3.1f} eV".format(te))
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Spectral Radiance [W/m^3/sr]")
ax.legend()

# plot influence of angle between polarisation and observation on scattered spectra
te, ao = 100, 90
_, ax = plt.subplots()
ax.set_title("Scattered spectrum, te = {:3.1f}, obs. angl = {:d}".format(te, ao))
for ap in angles_pol:
    rad = scattered[te][ap][ao]
    ax.plot(wvls, rad, label="pol. angl. = {:d} deg".format(ap))
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Spectral Radiance [W/m^3/sr]")
ax.legend()

# plot influence of observation angle on scattered spectra
te, ap = 2e3, 90
_, ax = plt.subplots()
ax.set_title("Scattered spectrum, te = {:3.1f}, obs. angl = {:d}".format(te, ao))
for ao in angles_obs:
    rad = scattered[te][ap][ao]
    ax.plot(wvls, rad, label="obs. angl. = {:d} deg".format(ao))
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("Spectral Radiance [W/m^3/sr]")
ax.legend()
plt.show()
