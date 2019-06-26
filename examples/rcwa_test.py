import numpy as np
import matplotlib.pyplot as plt

from solcore import si, material
from solcore.structure import Junction, Layer
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver, default_options
from solcore.light_source import LightSource
from rigorous_coupled_wave_analysis.rcwa import calculate_rat_rcwa
from solcore.absorption_calculator.nk_db import download_db, search_db
from solcore.material_system.create_new_material import create_new_material

GaAs = material('GaAs')()
Air = material('Air')()
TiO2 = material('TiO2', sopra=True)()  # for the nanoparticles
Ag = material('Ag_J')()
# Materials for the anti-reflection coating
MgF2 = material('MgF2')()
TiO2 = material('TiO2')()
Ta2O5 = material('410', nk_db = True)()
Ag = material('Ag_J')()

x = 500
# anti-reflection coating
ARC = [Layer(si('78nm'), material = MgF2), Layer(si('48nm'), material = Ta2O5)]

size = ((x, 0),(0,x))
# The layer with nanoparticles
struct_mirror = [Layer(si('120nm'), TiO2, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
                                                     'halfwidths': (210,210), 'angle': 0}])]
# NP_layer=[Layer(si('50nm'), Ag)]


solar_cell = SolarCell(ARC + [Layer(material=GaAs, width=si('300nm'))] + struct_mirror, substrate=Ag)

orders = 80

wavelengths = np.linspace(300, 1000, 50)*1e-9

options = {'nm_spacing': 0.5,
           'project_name': 'UC_PC',
           'calc_profile': False,
           'n_theta_bins': 100,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 1e-6, 'phi_in': 1e-6,
           'I_thresh': 0.001,
           'coherent': True,
           'coherency_list': None,
           'lookuptable_angles': 200,
           #'prof_layers': [1,2],
           'n_rays': 100000,
           'random_angles': False,
           'nx': 5, 'ny': 5,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }


output = calculate_rat_rcwa(solar_cell, size, orders, options, incidence=Air, substrate=Ag, only_incidence_angle=True,
                       front_or_rear='front')

plt.plot(wavelengths, output['A_layer'][:, 0, 0])
plt.plot(wavelengths, output['A_layer'][:, 0, 1])
plt.plot(wavelengths, output['A_layer'][:, 0, 2])
plt.plot(wavelengths, output['A_layer'][:, 0, 3])
plt.legend(['ARC1', 'ARC2', 'GaAs', 'struct mirror'])
plt.show()
