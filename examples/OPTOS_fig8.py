import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from structure import Interface, BulkLayer, Structure
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT
from angles import theta_summary
from textures.standard_rt_textures import regular_pyramids
from solcore.material_system import create_new_material
import matplotlib.pyplot as plt

import seaborn as sns
from cycler import cycler
#create_new_material('Si_OPTOS', 'data/Si_OPTOS_n.txt', 'data/Si_OPTOS_k.txt')
pal = sns.cubehelix_palette()

cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)
angle_degrees_in = 8

# matrix multiplication
wavelengths = np.linspace(900, 1200, 30)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'optos_checks',
           'calc_profile': False,
           'n_theta_bins': 100,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': angle_degrees_in*np.pi/180, 'phi_in': 1e-6,
           'I_thresh': 0.001,
           #'coherent': True,
           #'coherency_list': None,
           'lookuptable_angles': 200,
           #'prof_layers': [1,2],
           'n_rays': 100000,
           'random_angles': False,
           'nx': 5, 'ny': 5,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }


Si = material('Si_OPTOS')()
Air = material('Air')()

# materials with constant n, zero k
x = 1000

d_vectors = ((x, 0),(0,x))
area_fill_factor = 0.36
hw = np.sqrt(area_fill_factor)*500


front_materials = []
back_materials = []

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model
surf = regular_pyramids(elevation_angle=55, upright=False)


front_surf = Interface('RT_TMM', texture = surf, layers=[],
                       name = 'inv_pyramids' + str(options['n_rays']))
back_surf = Interface('RCWA', layers=back_materials, name = 'crossed_grating_60', d_vectors=d_vectors, rcwa_orders=60)
#back_surf = Interface('TMM', layers=[], name = 'planar_back', coherent=True)

bulk_Si = BulkLayer(200e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)

results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]

# load OPTOS/measured data

sim = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')

plt.figure()
plt.plot(wavelengths*1e9, RAT['R'][0])
plt.plot(wavelengths*1e9, RAT['T'][0])
plt.plot(wavelengths*1e9, RAT['A_bulk'][0], 'ko')
plt.plot(wavelengths*1e9, RAT['A_bulk'][0], 'k-')
plt.plot(sim[:,0], sim[:,1])
plt.plot(meas[:,0], meas[:,1])
plt.ylim([0, 1])
plt.legend(['R', 'T', 'A'])

plt.show()


from angles import make_angle_vector
from config import results_path
from sparse import load_npz


_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])

wl_to_plot = 1100e-9

wl_index = np.argmin(np.abs(wavelengths-wl_to_plot))

sprs = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'rearRT.npz'))

full = sprs[wl_index].todense()

summat = theta_summary(full, angle_vector, options['n_theta_bins'], 'rear')

summat_r = summat[options['n_theta_bins']:, :]

summat_r = summat_r.rename({r'$\theta_{in}$': r'$\sin(\theta_{in})$', r'$\theta_{out}$': r'$\sin(\theta_{out})$'})

summat_r = summat_r.assign_coords({r'$\sin(\theta_{in})$': np.sin(summat_r.coords[r'$\sin(\theta_{in})$']).data,
                                    r'$\sin(\theta_{out})$': np.sin(summat_r.coords[r'$\sin(\theta_{out})$']).data})

#ax = plt.subplot(212)

#ax = Tth.plot.imshow(ax=ax)


import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
fig = plt.figure(figsize=(10,3.5))
ax = fig.add_subplot(1,2,1, aspect='equal')
ax = summat_r.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
#fig.savefig('perovskite_Si_frontsurf_rearR.pdf', bbox_inches='tight', format='pdf')
#ax = Tth.plot.imshow(ax=ax)

#plt.show()



sprs = load_npz(os.path.join(results_path, options['project_name'], SC[2].name + 'frontRT.npz'))

full = sprs[wl_index].todense()

summat = theta_summary(full, angle_vector, options['n_theta_bins'], 'front')

summat_r = summat[:options['n_theta_bins'], :]

summat_r = summat_r.rename({r'$\theta_{in}$': r'$\sin(\theta_{in})$', r'$\theta_{out}$': r'$\sin(\theta_{out})$'})

summat_r = summat_r.assign_coords({r'$\sin(\theta_{in})$': np.sin(summat_r.coords[r'$\sin(\theta_{in})$']).data,
                                    r'$\sin(\theta_{out})$': np.sin(summat_r.coords[r'$\sin(\theta_{out})$']).data})




import matplotlib as mpl
palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)
#fig = plt.figure(figsize=(5,4))
ax2 = fig.add_subplot(1,2,2, aspect='equal')
ax2 = summat_r.plot.imshow(ax=ax2, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('perovskite_Si_frontsurf_rearRT.pdf', bbox_inches='tight', format='pdf')
#ax = Tth.plot.imshow(ax=ax)

plt.show()

