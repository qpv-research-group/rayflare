import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism.process_structure import process_structure
from rayflare.matrix_formalism.multiply_matrices import calculate_RAT
from rayflare.angles import theta_summary
from rayflare.textures.standard_rt_textures import regular_pyramids, V_grooves
from solcore.material_system import create_new_material
import matplotlib.pyplot as plt
from rayflare.angles import make_angle_vector
from rayflare.config import results_path
from sparse import load_npz



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
wavelengths = np.arange(800, 1200, 5)*1e-9
options = {'depth_spacing': 0.5,
           'project_name': 'pyramids_lambertian',
           'calc_profile': False,
           'n_theta_bins': 50,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': angle_degrees_in*np.pi/180, 'phi_in': 1e-6,
           'I_thresh': 0.001,
           #'coherent': True,
           #'coherency_list': None,
           'lookuptable_angles': 200,
           #'prof_layers': [1,2],
           'n_rays': 1500,
           'random_angles': False,
           'nx': 21, 'ny': 21,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True,
           'random_ray_position': False,
           'randomize': False,
           'avoid_edges': False
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
surf = regular_pyramids(elevation_angle=55, size=5, upright=True)


front_surf = Interface('RT_TMM', texture = surf, layers=[Layer(si('0.1nm'), Air)],
                       name = 'pyramids' + str(options['n_rays']))
back_surf = Interface('Lambertian', layers=[], name = 'lambertian')

bulk_Si = BulkLayer(201.8e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Air)

process_structure(SC, options)
results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]

# load OPTOS/measured data

sim = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')

plt.figure()
plt.plot(wavelengths*1e9, RAT['R'][0], label='R')
plt.plot(wavelengths*1e9, RAT['T'][0], label='T')
plt.plot(wavelengths*1e9, RAT['A_bulk'][0], 'ko-')
plt.plot(sim[:,0], sim[:,1], label='inv pyr/planar')
plt.ylim([0, 1])

plt.legend()
plt.show()


R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]

np.savetxt('lambertianpyramids_rayflare.txt', np.vstack((RAT['A_bulk'][0], R_0)).T)

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])
#
# wl_to_plot = 1100e-9
#
# wl_index = np.argmin(np.abs(wavelengths-wl_to_plot))
#
# sprs = load_npz(os.path.join(results_path, options['project_name'], SC[0].name + 'rearRT.npz'))
#
# full = sprs[wl_index].todense()
#
# summat = theta_summary(full, angle_vector, options['n_theta_bins'], 'rear')
#
# summat_r = summat[options['n_theta_bins']:, :]
#
# summat_r= summat_r.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})
#
# summat_r= summat_r.assign_coords(a=np.sin(summat_r.coords['a']).data,
#                                                   b=np.sin(summat_r.coords['b']).data)
#
#
# summat_r= summat_r.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})
#
# #ax = plt.subplot(212)
#
# #ax = Tth.plot.imshow(ax=ax)
#
#
# import matplotlib as mpl
# palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
# palhf.reverse()
# seamap = mpl.colors.ListedColormap(palhf)
# fig = plt.figure(figsize=(10,3.5))
# ax = fig.add_subplot(1,2,1, aspect='equal')
# ax.text(-0.15, 1, 'a)')
# ax = summat_r.plot.imshow(ax=ax, cmap=seamap)
#
# #ax = plt.subplot(212)
# #fig.savefig('perovskite_Si_frontsurf_rearR.pdf', bbox_inches='tight', format='pdf')
# #ax = Tth.plot.imshow(ax=ax)
#
# #plt.show()
#
#
#
# sprs = load_npz(os.path.join(results_path, options['project_name'], SC[2].name + 'frontRT.npz'))
#
# full = sprs[wl_index].todense()
#
# summat = theta_summary(full, angle_vector, options['n_theta_bins'], 'front')
#
# summat_r = summat[:options['n_theta_bins'], :]
#
#
# summat_r= summat_r.rename({r'$\theta_{in}$': 'a', r'$\theta_{out}$': 'b'})
#
# summat_r= summat_r.assign_coords(a=np.sin(summat_r.coords['a']).data,
#                                                   b=np.sin(summat_r.coords['b']).data)
#
#
# summat_r= summat_r.rename({'a': r'$\sin(\theta_{in})$', 'b': r'$\sin(\theta_{out})$'})
#
#
#
#
# import matplotlib as mpl
# palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
# palhf.reverse()
# seamap = mpl.colors.ListedColormap(palhf)
# #fig = plt.figure(figsize=(5,4))
# ax2 = fig.add_subplot(1,2,2, aspect='equal')
# ax2.text(-0.15, 1, 'b)')
# ax2 = summat_r.plot.imshow(ax=ax2, cmap=seamap)
#
# #ax = plt.subplot(212)
# fig.savefig('optos_comparison_matrices.pdf', bbox_inches='tight', format='pdf')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()

