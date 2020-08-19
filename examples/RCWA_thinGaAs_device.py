import numpy as np
from solcore import si, material
from solcore.structure import Layer, Junction
from rigorous_coupled_wave_analysis.rcwa import rcwa_structure
from solcore.solar_cell_solver import solar_cell_solver
from solcore.light_source import LightSource
import matplotlib.pyplot as plt

# Import the DE implementations
from solcore.optimization import DE

from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell
from solcore.state import State

import seaborn as sns
from cycler import cycler




GaAs = material('GaAs')

InAlP = material('AlInP')
InGaP = material('GaInP')

Ag = material('Ag')()

SiN = material('SiN_SE')()


x = 500

# anti-reflection coating


size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

wavelengths = np.linspace(250, 930, 140) * 1e-9

RCWA_wl = wavelengths

options = {'nm_spacing': 0.5,
           'n_theta_bins': 100,
           'c_azimuth': 1e-7,
           'pol': 's',
           'wavelengths': RCWA_wl,
           'theta_in': 0, 'phi_in': 0,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi / 2,
           'project_name': 'ultrathin'
           }

ropt = dict(LatticeTruncation='Circular',
            DiscretizedEpsilon=True,
            DiscretizationResolution=4,
            PolarizationDecomposition=False,
            PolarizationBasis='Default',
            LanczosSmoothing=True,
            SubpixelSmoothing=True,
            ConserveMemory=False,
            WeismannFormulation=True)

light_source = LightSource(source_type='standard', version='AM1.5g')


options = State()
options['rcwa_options'] = ropt
options.optics_method = 'RCWA'
options.wavelength = wavelengths
options.light_source = light_source
options.pol = 's'
options.mpp = True
options.light_iv = True
options.position = 1e-10
options.voltages = np.linspace(-1.5, 1.5, 100)
options.size = size
options.orders = 37

window_material = InGaP(In=0.49, Nd = si(5e18, 'cm-3'))

top_cell_n_material = GaAs(Nd=si("1e18cm-3"))
top_cell_p_material = GaAs(Na=si("1e18cm-3"))

bsf_material = InAlP(Al=0.53)

grating1 = [Layer(si(30, 'nm'), SiN)]
grating2 = [Layer(si(77, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                'radius': 105, 'angle': 0}])]

solar_cell = SolarCell([Junction([Layer(material=window_material, width=si('19nm'), role='window'),
                               Layer(material=top_cell_p_material, width=si('43nm'), kind='emitter'),
                                Layer(material=top_cell_n_material, width=si('43nm'), kind='base'),
                               Layer(material=bsf_material, width=si('18nm'))], kind='PDD')] + grating1 + grating2)

solar_cell_solver(solar_cell, 'qe', user_options=options)

solar_cell_solver(solar_cell, 'iv', user_options=options)

EQE_data = np.loadtxt('data/V2A_9_EQE.dat', skiprows=129)

plt.figure()
plt.plot(wavelengths * 1e9, solar_cell.absorbed * 100, 'k--', label='Absorption')
plt.plot(wavelengths * 1e9, solar_cell[0].layer_absorption * 100, 'r-', label='Window')
plt.plot(wavelengths * 1e9, solar_cell[1].layer_absorption * 100, label='Grating 1')
plt.plot(wavelengths * 1e9, solar_cell[2].layer_absorption * 100, label='Grating 2')
plt.plot(wavelengths * 1e9, solar_cell[0].eqe(wavelengths) * 100, label='EQE')
plt.plot(EQE_data[:,0], EQE_data[:,1], label='EQE meas')
plt.legend(loc='upper right')
plt.ylim(0, 100)
plt.ylabel('EQE (%)')
plt.xlabel('Wavelength (nm)')
plt.text(280-(1850-280)*0.12, 100, 'a)')
plt.show()

