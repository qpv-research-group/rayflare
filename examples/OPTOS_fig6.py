import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT

import matplotlib.pyplot as plt

# matrix multiplication
wavelengths = np.linspace(1000, 1200, 20)*1e-9
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

# materials with constant n, zero k
Spiro = [12e-9, np.array([0,1]), np.array([1.65, 1.65]), np.array([0,0])]
SnO2 = [10e-9, np.array([0,1]), np.array([2, 2]), np.array([0,0])]

x = 1000

d_vectors = ((x, 0),(0,x))
area_fill_factor = 0.36
hw = np.sqrt(area_fill_factor)*500

front_materials = []
back_materials = [Layer(si('120nm'), Si, geometry=[{'type': 'rectangle', 'mat': Air, 'center': (x/2, x/2),
                                                     'halfwidths': (hw, hw), 'angle': 45}])]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model


front_surf = Interface('TMM', layers=front_materials, name = 'planar', coherent=True)
back_surf = Interface('RCWA', layers=back_materials, name = 'crossed_grating', d_vectors=d_vectors, rcwa_orders=20)
#back_surf = Interface('TMM', layers=[], name = 'planar_back', coherent=True)

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

plt.show()
from angles import make_angle_vector
from config import results_path
from sparse import load_npz

#_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
#                                       options['c_azimuth'])

#sprs = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'frontRT.npz'))

#full = sprs[15].todense()


#summat, Rsum, Tsum = theta_summary(full, angle_vector)

#Rth = summat[0:100,:]
#Tth = summat[100:, :]
#Rth = xr.DataArray(Rth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'], coords={r'$\sin(\theta_{out})$': np.linspace(0,1,100),
#                                                                            r'$\sin(\theta_{in})$': np.linspace(0,1,100)})
#Tth = xr.DataArray(Tth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'], coords={r'$\sin(\theta_{out})$': np.linspace(0,1,100),
#                                                                            r'$\sin(\theta_{in})$': np.linspace(0,1,100)})

#import matplotlib as mpl
#palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
#palhf.reverse()
#seamap = mpl.colors.ListedColormap(palhf)
#fig = plt.figure()
#ax = plt.subplot(111)
#ax = Rth.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)

#ax = Tth.plot.imshow(ax=ax)

_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])

sprs = load_npz(os.path.join(results_path, options['project_name'], SC[2].name + 'frontRT.npz'))

full = sprs[15].todense()

plt.figure()
plt.imshow(full)
plt.colorbar()

plt.show()

