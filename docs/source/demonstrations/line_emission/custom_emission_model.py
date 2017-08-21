
# Core and external imports
from math import sqrt, exp

from pylab import ion
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.observer.imaging.vector import VectorCamera
from scipy.constants import elementary_charge, speed_of_light, pi
from tools.jet.calcam import load_calcam_calibration
from tools.simulation_data.solps.solps_plasma import SOLPSSimulation

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic import deuterium
from cherab.core.atomic import get_adf15_pec
from raysect.optical import World, translate
from raysect.optical.material.emitter import VolumeEmitterInhomogeneous
from raysect.primitive import Cylinder, import_stl

# Define constants
PI = pi
AMU = 1.66053892e-27
ELEMENTARY_CHARGE = elementary_charge
SPEED_OF_LIGHT = speed_of_light

world = World()


MESH_PARTS = ['/projects/cadmesh/mast/mastu-light/mug_centrecolumn_endplates.stl',
             '/projects/cadmesh/mast/mastu-light/mug_divertor_nose_plates.stl']

for path in MESH_PARTS:
   import_stl(path, parent=world, material=AbsorbingSurface())  # Mesh with perfect absorber

# Load plasma from SOLPS model
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 69636
sim = SOLPSSimulation.load_from_mdsplus(mds_server, ref_number)
plasma = sim.plasma
mesh = sim.mesh
vessel = mesh.vessel


class ExcitationLine(VolumeEmitterInhomogeneous):

    def __init__(self, line, electron_distribution, atom_species, step=0.005, block=0, filename=None):

        super().__init__(step=step)

        self.line = line
        self.electron_distribution = electron_distribution
        self.atom_species = atom_species

        # Function to load ADAS rate coefficients
        # Replace this with your own atomic data as necessary.
        self.pec_excitation = get_adf15_pec(line, 'EXCIT', filename=filename, block=block)

    def emission_function(self, point, direction, spectrum, world, ray, primitive, to_local, to_world):

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

        # pec for excitation at this temperature and density
        pece = self.pec_excitation(ne, te)

        # calculate line intensity
        inty = 1E6 * (pece * ne * na)

        # Calculate simple gaussian line at each line wavelength in spectrum
        # Add it to the existing spectrum. Don't override previous results!
        weight = self.line.element.atomic_weight
        rest_wavelength = self.line.wavelength
        sigma = sqrt(te * ELEMENTARY_CHARGE / (weight * AMU)) * rest_wavelength / SPEED_OF_LIGHT
        i0 = inty/(sigma * sqrt(2 * PI))
        width = 2*sigma**2
        for i, wvl in enumerate(spectrum.wavelengths):
            spectrum.samples[i] += i0 * exp(-(wvl - rest_wavelength)**2 / width)

        return spectrum


# Setup deuterium line
d_alpha = Line(deuterium, 0, (3, 2), wavelength=656.19)

# Load the deuterium atom species and electron distribution for use in rate calculations.
d_atom_species = plasma.get_species(deuterium, 0)
electrons = plasma.electron_distribution

# Load the Excitation and Recombination lines and add them as emitters to the world.
d_alpha_excit = ExcitationLine(d_alpha, plasma.electron_distribution, d_atom_species)

outer_radius = plasma.misc_properties['maxr'] + 0.001
plasma_height = plasma.misc_properties['maxz'] - plasma.misc_properties['minz']
lower_z = plasma.misc_properties['minz']

main_plasma_cylinder = Cylinder(outer_radius, plasma_height, parent=world,
                                material=d_alpha_excit, transform=translate(0, 0, lower_z))

# Load a MAST-U midplane camera
camera_config = load_calcam_calibration('./demo/mast/camera_configs/mug_bulletb_midplane.nc')

# Setup camera for interactive use...
pixels_shape, pixel_origins, pixel_directions = camera_config
camera = VectorCamera(pixel_origins, pixel_directions, pixels=pixels_shape, parent=world)
camera.spectral_samples = 15
camera.pixel_samples = 1
camera.display_progress = True
camera.display_update_time = 20
ion()
camera.observe()



