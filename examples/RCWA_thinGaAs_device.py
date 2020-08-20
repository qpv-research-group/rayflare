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




GaAs = material('GaAs_WVASE')

InAlP = material('InAlP_WVASE')
InGaP = material('InGaP_WVASE')

Ag = material('Ag_Jiang')()

SiN = material('SiN_SE')()


x = 500

# anti-reflection coating


size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

wavelengths = np.linspace(250, 930, 140) * 1e-9

RCWA_wl = wavelengths



ropt = dict(LatticeTruncation='Circular',
            DiscretizedEpsilon=True,
            DiscretizationResolution=4,
            PolarizationDecomposition=False,
            PolarizationBasis='Default',
            LanczosSmoothing=True,
            SubpixelSmoothing=True,
            ConserveMemory=False,
            WeismannFormulation=True)

light_source = LightSource(source_type='standard', version='AM0')


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
options.parallel = True

window_material = InGaP(In=0.49, Nd = si(5e18, 'cm-3'))

top_cell_n_material = GaAs(Nd=si("1e18cm-3"))
top_cell_p_material = GaAs(Na=si("1e18cm-3"))

bsf_material = InAlP(Al=0.53)

grating1 = [Layer(si(28, 'nm'), SiN)]
grating2 = [Layer(si(79, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                'radius': 105, 'angle': 0}])]
#
# solar_cell = SolarCell([Junction([Layer(material=window_material, width=si('19nm'), role='window'),
#                                Layer(material=top_cell_p_material, width=si('44.5nm'), kind='emitter'),
#                                 Layer(material=top_cell_n_material, width=si('44.5nm'), kind='base'),
#                                Layer(material=bsf_material, width=si('18nm'))], kind='PDD')] + grating1 + grating2)

# solar_cell_solver(solar_cell, 'qe', user_options=options)
#
# solar_cell_solver(solar_cell, 'iv', user_options=options)

EQE_data = np.loadtxt('data/V2A_9_EQE.dat', skiprows=129)




solar_cell_optical = SolarCell([Layer(material=window_material, width=si('19nm')),
                               Layer(material=top_cell_p_material, width=si('44.5nm')),
                                Layer(material=top_cell_n_material, width=si('44.5nm')),
                               Layer(material=bsf_material, width=si('18nm'))] + grating1 + grating2, substrate=Ag)

solar_cell_solver(solar_cell_optical, 'optics', user_options=options)

plt.figure()
# plt.plot(wavelengths * 1e9, solar_cell[0].layer_absorption * 100, 'r-', label='III-V')
# plt.plot(wavelengths * 1e9, (solar_cell[1].layer_absorption + solar_cell[2].layer_absorption) * 100, label='Grating')
# plt.plot(wavelengths * 1e9, solar_cell[0].eqe(wavelengths) * 100, label='EQE')
plt.plot(wavelengths *1e9,
         (solar_cell_optical[0].layer_absorption +
          solar_cell_optical[1].layer_absorption +
          solar_cell_optical[2].layer_absorption)*100,
         label='GaInP + GaAs')
plt.plot(EQE_data[:,0], EQE_data[:,1], '--', label='EQE meas')
plt.legend(loc='upper right')
plt.ylim(0, 100)
plt.ylabel('EQE (%)')
plt.xlabel('Wavelength (nm)')
plt.text(280-(1850-280)*0.12, 100, 'a)')
plt.show()
#
#
# shape_mats = []
# geom_list_str = [None] * len(gl)
# for i1, geom in enumerate(gl):
#     if bool(geom):
#         shape_mats.append([x['mat'] for x in geom])
#         geom_list_str[i1] = [{}] * len(geom)
#         for i2, g in enumerate(geom):
#             for item in g.keys():
#                 print(item)
#                 if item != 'mat':
#                     geom_list_str[i1][i2][item] = deepcopy(gl[i1][i2][item])
#                 else:
#                     geom_list_str[i1][i2][item] = str(gl[i1][i2][item])
#


GaAs = material('GaAs_WVASE')()
Air = material('Air')()

InAlP = material('InAlP_WVASE')()
InGaP = material('InGaP_WVASE')()

Ag = material('Ag_Jiang')()

# MgF2 = material('MgF2')()
# Ta2O5 = material('410', nk_db=True)()
SiN = material('SiN_SE')()

solar_cell = SolarCell([Layer(material=window_material, width=si('19nm')),
                        Layer(material=top_cell_p_material, width=si(89, 'nm')),
                        Layer(material=bsf_material, width=si('18nm'))] + grating1 + grating2,
                       substrate=Ag)

options = {'nm_spacing': 0.5, 'n_theta_bins': 100, 'c_azimuth': 1e-7,
           'pol': 's',
           'wavelengths': wavelengths,
           'theta_in': 0, 'phi_in': 0,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi / 2,
           'project_name': 'ultrathin'
           }


options['rcwa_options'] = ropt

S4_setup = rcwa_structure(solar_cell, size, 37, options, Air, Ag)

RAT = S4_setup.calculate()

plt.figure()
# plt.plot(wavelengths * 1e9, solar_cell[0].layer_absorption * 100, 'r-', label='III-V')
# plt.plot(wavelengths * 1e9, (solar_cell[1].layer_absorption + solar_cell[2].layer_absorption) * 100, label='Grating')
# plt.plot(wavelengths * 1e9, solar_cell[0].eqe(wavelengths) * 100, label='EQE')
plt.plot(wavelengths *1e9,
         (RAT['A_layer'][:,0] + RAT['A_layer'][:,1])*100,
         label='GaInP + GaAs')
plt.plot(EQE_data[:,0], EQE_data[:,1], '--', label='EQE meas')
plt.legend(loc='upper right')
plt.ylim(0, 100)
plt.ylabel('EQE (%)')
plt.xlabel('Wavelength (nm)')
plt.text(280-(1850-280)*0.12, 100, 'a)')
plt.show()