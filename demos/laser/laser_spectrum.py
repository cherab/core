from cherab.core.model.laser import ConstantSpectrum, GaussianSpectrum

import matplotlib.pyplot as plt


# construct a ConstantSpectrum with 10 spectral bins
constant_wide = ConstantSpectrum(min_wavelength=1059.9, max_wavelength=1060.1, bins=10)

# plot the power_spectral_density attribute of the laser
_, ax = plt.subplots()
ax.plot(constant_wide.wavelengths, constant_wide.power_spectral_density)

ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("W / nm")

ax.set_title("Energy Spectral Density")
plt.show()

# construct a narrow laser spectrum
constant_narrow = ConstantSpectrum(min_wavelength=1059.999, max_wavelength=1060.001, bins=1)
print("narow spectrum wavelengths: {}, power spectral density: {}".format(constant_narrow.wavelengths,
                                                                          constant_narrow.power_spectral_density))

# construct a GaussianSpectrum with 20 bins
gaussian = GaussianSpectrum(min_wavelength=1059, max_wavelength=1061, bins=30,
                            mean=1060, stddev=0.3)

_, ax = plt.subplots()
ax.plot(gaussian.wavelengths, gaussian.power_spectral_density)
ax.set_xlabel("Wavelength [nm]")
ax.set_ylabel("W / nm")

ax.set_title("Energy Spectral Density")
plt.show()
