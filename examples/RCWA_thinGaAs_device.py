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



GaAs_nk = material('GaAs_WVASE')()


InAlP_nk = material('InAlP_WVASE')()
InGaP_nk = material('InGaP_WVASE')()

GaAs = material('GaAs')

InAlP = material('AlInP')
InGaP = material('GaInP')

Ag = material('Ag_Jiang')()

SiN = material('SiN_SE')()


x = 500

# anti-reflection coating


size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

wavelengths = np.linspace(250, 1100, 80) * 1e-9

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
options.orders = 20
options.parallel = True

window_material = InGaP(In=0.485, Na = si(5e18, 'cm-3'))
# window_material.n_data = np.stack([wavelengths, InGaP_nk.n(wavelengths)])
# window_material.k_data = np.stack([wavelengths, InGaP_nk.k(wavelengths)])
# window_material.n_path = '/home/phoebe/Documents/rayflare/examples/data/InGaP_n.txt'
# window_material.k_path = '/home/phoebe/Documents/rayflare/examples/data/InGaP_k.txt'

to_save_for_WVASE = np.stack([wavelengths*1e9, window_material.n(wavelengths), window_material.k(wavelengths)]).T
np.savetxt('InGaP_interp_Solcore.mat', to_save_for_WVASE)

top_cell_n_material = GaAs(Nd=si("1e18cm-3"))
top_cell_p_material = GaAs(Na=si("1e18cm-3"))

# top_cell_n_material.n_data = np.stack([wavelengths, GaAs_nk.n(wavelengths)])
# top_cell_n_material.k_data = np.stack([wavelengths, GaAs_nk.k(wavelengths)])
# top_cell_p_material.n_data = np.stack([wavelengths, GaAs_nk.n(wavelengths)])
# top_cell_p_material.k_data = np.stack([wavelengths, GaAs_nk.k(wavelengths)])
# top_cell_n_material.n_path = '/home/phoebe/Documents/rayflare/examples/data/GaAs_subs_n.txt'
# top_cell_n_material.k_path = '/home/phoebe/Documents/rayflare/examples/data/GaAs_subs_k.txt'
# top_cell_p_material.n_path = '/home/phoebe/Documents/rayflare/examples/data/GaAs_subs_n.txt'
# top_cell_p_material.k_path = '/home/phoebe/Documents/rayflare/examples/data/GaAs_subs_k.txt'

bsf_material = InAlP(Al=0.535, Nd=si("5e18cm-3"))
# bsf_material.n_data = np.stack([wavelengths, InAlP_nk.n(wavelengths)])
# bsf_material.k_data = np.stack([wavelengths, InAlP_nk.k(wavelengths)])
# bsf_material.n_path = '/home/phoebe/Documents/rayflare/examples/data/InAlP_n.txt'
# bsf_material.k_path = '/home/phoebe/Documents/rayflare/examples/data/InAlP_k.txt'


pal = sns.color_palette("husl", 3)
wavelengths_plot = np.linspace(250, 1000, 600) * 1e-9

plt.figure()
plt.plot(wavelengths_plot*1e9, InGaP_nk.n(wavelengths_plot), label='InGaP WVASE', color=pal[0])
plt.plot(wavelengths_plot*1e9, window_material.n(wavelengths_plot), '--', label='InGaP Solcore', color=pal[0])

plt.plot(wavelengths_plot*1e9, InAlP_nk.n(wavelengths_plot), label='InAlP WVASE', color=pal[1])
plt.plot(wavelengths_plot*1e9, bsf_material.n(wavelengths_plot), '--', label='InAlP Solcore', color=pal[1])

plt.plot(wavelengths_plot*1e9, GaAs_nk.n(wavelengths_plot), label='GaAs WVASE', color=pal[2])
plt.plot(wavelengths_plot*1e9, top_cell_n_material.n(wavelengths_plot), '--', label='GaAs Solcore', color=pal[2])
plt.legend()
plt.show()

plt.figure()
plt.plot(wavelengths_plot*1e9, InGaP_nk.k(wavelengths_plot), label='InGaP WVASE', color=pal[0])
plt.plot(wavelengths_plot*1e9, window_material.k(wavelengths_plot), '--',  label='InGaP Solcore', color=pal[0])

plt.plot(wavelengths_plot*1e9, InAlP_nk.k(wavelengths_plot), label='InAlP WVASE', color=pal[1])
plt.plot(wavelengths_plot*1e9, bsf_material.k(wavelengths_plot), '--',  label='InAlP Solcore', color=pal[1])

plt.plot(wavelengths_plot*1e9, GaAs_nk.k(wavelengths_plot), label='GaAs WVASE', color=pal[2])
plt.plot(wavelengths_plot*1e9, top_cell_n_material.k(wavelengths_plot), '--', label='GaAs Solcore', color=pal[2])
plt.legend()
plt.show()


grating1 = [Layer(si(28, 'nm'), SiN)]
grating2 = [Layer(si(79, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                'radius': 110, 'angle': 0}])]

solar_cell = SolarCell([Junction([Layer(material=window_material, width=si('19nm'), role='window'),
                               Layer(material=top_cell_p_material, width=si('43.5nm'), kind='emitter'),
                                Layer(material=top_cell_n_material, width=si('43.5nm'), kind='base'),
                               Layer(material=bsf_material, width=si('18nm'))], kind='PDD')] + grating1 + grating2,
                       substrate=Ag)

solar_cell_solver(solar_cell, 'qe', user_options=options)

#solar_cell_solver(solar_cell, 'iv', user_options=options)

EQE_data = np.loadtxt('data/V2A_9_EQE.dat', skiprows=129)




solar_cell_optical = SolarCell([Layer(material=InGaP_nk, width=si('19nm')),
                                Layer(material=GaAs_nk, width=si('43.5nm')),
                                Layer(material=GaAs_nk, width=si('43.5nm')),
                                Layer(material=InAlP_nk, width=si('18nm'))] + grating1 + grating2, substrate=Ag)

solar_cell_solver(solar_cell_optical, 'optics', user_options=options)


solar_cell_planar = SolarCell([Junction([Layer(material=window_material, width=si('19nm'), role='window'),
                               Layer(material=top_cell_p_material, width=si('43.5nm'), kind='emitter'),
                                Layer(material=top_cell_n_material, width=si('43.5nm'), kind='base'),
                               Layer(material=bsf_material, width=si('18nm'))], kind='PDD')] +
                              [Layer(material=GaAs_nk, width=si('300nm'))], substrate=Ag)

options.optics_method = 'TMM'


solar_cell_solver(solar_cell_planar, 'qe', user_options=options)

planar_fr = 0.03
weighted_EQE = 0.9*((1-planar_fr)*solar_cell[0].eqe(wavelengths) + planar_fr*solar_cell_planar[0].eqe(wavelengths))

plt.figure()
plt.plot(wavelengths * 1e9, (solar_cell[1].layer_absorption + solar_cell[2].layer_absorption) * 100,
         label='Grating absorption')

plt.plot(wavelengths * 1e9, 0.9*(solar_cell[0].layer_absorption) * 100,
         label='III-V absorption')
plt.plot(wavelengths * 1e9, 0.9*solar_cell[0].eqe(wavelengths) * 100, '--', label='EQE')
plt.plot(wavelengths * 1e9, weighted_EQE*100, '--', label='weighted')
plt.plot(wavelengths*1e9, 0.9*solar_cell_planar[0].eqe(wavelengths)*100, label='contact EQE')
plt.plot(wavelengths *1e9,
          0.9*(solar_cell_optical[0].layer_absorption +
           solar_cell_optical[1].layer_absorption +
           solar_cell_optical[2].layer_absorption)*100,
          label='GaInP + GaAs optical (SE data)')
plt.plot(EQE_data[:,0], EQE_data[:,1], '--', label='EQE meas')
plt.legend(loc='upper right')

plt.ylabel('EQE (%)')
plt.xlabel('Wavelength (nm)')
plt.show()


# #
# #
# # shape_mats = []
# # geom_list_str = [None] * len(gl)
# # for i1, geom in enumerate(gl):
# #     if bool(geom):
# #         shape_mats.append([x['mat'] for x in geom])
# #         geom_list_str[i1] = [{}] * len(geom)
# #         for i2, g in enumerate(geom):
# #             for item in g.keys():
# #                 print(item)
# #                 if item != 'mat':
# #                     geom_list_str[i1][i2][item] = deepcopy(gl[i1][i2][item])
# #                 else:
# #                     geom_list_str[i1][i2][item] = str(gl[i1][i2][item])
# #
#
#
# GaAs = material('GaAs_WVASE')()
# Air = material('Air')()
#
# InAlP = material('InAlP_WVASE')()
# InGaP = material('InGaP_WVASE')()
#
# Ag = material('Ag_Jiang')()
#
# # MgF2 = material('MgF2')()
# # Ta2O5 = material('410', nk_db=True)()
# SiN = material('SiN_SE')()
#
# solar_cell = SolarCell([Layer(material=window_material, width=si('19nm')),
#                         Layer(material=top_cell_p_material, width=si(89, 'nm')),
#                         Layer(material=bsf_material, width=si('18nm'))] + grating1 + grating2,
#                        substrate=Ag)
#
# options = {'nm_spacing': 0.5, 'n_theta_bins': 100, 'c_azimuth': 1e-7, 'pol': 's',
#            'wavelengths': wavelengths,
#            'theta_in': 0, 'phi_in': 0,
#            'parallel': True, 'n_jobs': -1,
#            'phi_symmetry': np.pi / 2,
#            'project_name': 'ultrathin'
#            }
#
#
# options['rcwa_options'] = ropt
#
# S4_setup = rcwa_structure(solar_cell, size, 37, options, Air, Ag)
#
# RAT = S4_setup.calculate()
#
# plt.figure()
# # plt.plot(wavelengths * 1e9, solar_cell[0].layer_absorption * 100, 'r-', label='III-V')
# # plt.plot(wavelengths * 1e9, (solar_cell[1].layer_absorption + solar_cell[2].layer_absorption) * 100, label='Grating')
# # plt.plot(wavelengths * 1e9, solar_cell[0].eqe(wavelengths) * 100, label='EQE')
# plt.plot(wavelengths *1e9,
#          (RAT['A_layer'][:,0] + RAT['A_layer'][:,1])*100,
#          label='GaInP + GaAs')
# plt.plot(EQE_data[:,0], EQE_data[:,1], '--', label='EQE meas')
# plt.legend(loc='upper right')
# plt.ylim(0, 100)
# plt.ylabel('EQE (%)')
# plt.xlabel('Wavelength (nm)')
# plt.text(280-(1850-280)*0.12, 100, 'a)')
# plt.show()