import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q
from solcore.material_system.create_new_material import create_new_material
from solcore import si
from textures.standard_rt_textures import regular_pyramids
from structure import Interface, BulkLayer, Structure
from process_structure import process_structure, calculate_RAT

import matplotlib.pyplot as plt
import seaborn as sns

# matrix multiplication
wavelengths = np.linspace(900, 1200, 20)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'tmm_rcwa',
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


Si = material('Si')()
Air = material('Air')()
MgF2 = material('MgF2_RdeM')()
ITO_back = material('ITO_lowdoping')()
Perovskite = material('Perovskite_CsBr')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()
LiF = material('LiF')()
IZO = material('IZO')()
C60 = material('C60')()

# materials with constant n, zero k
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

x = 1000

d_vectors = ((x, 0),(0,x))

front_materials = []
back_materials = [Layer(si('120nm'), Si, geometry=[{'type': 'rectangle', 'mat': Air, 'center': (x/2, x/2),
                                                     'halfwidths': (np.sqrt(2*(500**2))/2, np.sqrt(2*(500**2))/2), 'angle': 45}])]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model


front_surf = Interface('TMM', layers=front_materials, name = 'planar', coherent=True)
back_surf = Interface('RCWA', layers=back_materials, name = 'crossed_grating', d_vectors=d_vectors, rcwa_orders=50)
back_surf = Interface('TMM', layers=[], name = 'planar_back', coherent=True)

bulk_Si = BulkLayer(200e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]

plt.figure()
plt.plot(wavelengths*1e9, RAT['R'][0])
plt.plot(wavelengths*1e9, RAT['T'][0])
plt.plot(wavelengths*1e9, RAT['A_bulk'][0])
plt.legend(['R', 'T', 'A'])
