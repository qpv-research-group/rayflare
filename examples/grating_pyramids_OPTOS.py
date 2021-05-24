import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import process_structure, calculate_RAT, get_savepath
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.angles import theta_summary, make_angle_vector
from rayflare.textures import regular_pyramids
from rayflare.options import default_options

import matplotlib.pyplot as plt
from sparse import load_npz
import seaborn as sns
import matplotlib as mpl

# Needs to run only once per computer to add material to database:
# from solcore.material_system import create_new_material
# create_new_material('Si_OPTOS', 'data/Si_OPTOS_n.txt', 'data/Si_OPTOS_k.txt')

angle_degrees_in = 8

wavelengths = np.linspace(900, 1200, 30)*1e-9

Si = material('Si_OPTOS')()
Air = material('Air')()

options = default_options()
options.wavelengths = wavelengths
options.theta_in = angle_degrees_in*np.pi/180
options.n_theta_bins = 100
options.c_azimuth = 0.25
options.n_rays = 25*25*1300
options.project_name = 'OPTOS_comparison'
options.phi_symmetry = np.pi/2
options.I_thresh = 1e-3
options.pol = 'u'
options.nx = 25
options.ny = 25
options.parallel = True

# materials with constant n, zero k
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
surf = regular_pyramids(elevation_angle=55, upright=False)

front_surf_pyramids = Interface('RT_Fresnel', texture = surf, layers=[],
                                name = 'inv_pyramids_front_' + str(options['n_rays']))
front_surf_planar = Interface('TMM', layers=[], name='planar_front')
back_surf_grating = Interface('RCWA', layers=back_materials, name='crossed_grating_back',
                              d_vectors=d_vectors, rcwa_orders=15)
back_surf_planar = Interface('TMM', layers=[], name = 'planar_back')

bulk_Si = BulkLayer(201.8e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC_fig6 = Structure([front_surf_planar, bulk_Si, back_surf_grating], incidence=Air, transmission=Air)

SC_fig7 = Structure([front_surf_pyramids, bulk_Si, back_surf_planar], incidence=Air, transmission=Air)

SC_fig8 = Structure([front_surf_pyramids, bulk_Si, back_surf_grating], incidence=Air, transmission=Air)

planar = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=Air)

process_structure(SC_fig6, options)

process_structure(SC_fig7, options)

process_structure(SC_fig8, options)

results_fig6= calculate_RAT(SC_fig6, options)

results_fig7 = calculate_RAT(SC_fig7, options)

results_fig8 = calculate_RAT(SC_fig8, options)

results_planar = calculate_RAT(planar, options)

RAT_fig6 = results_fig6[0]
RAT_fig7 = results_fig7[0]
RAT_fig8 = results_fig8[0]

RAT_planar = results_planar[0]

sim_fig6 = np.loadtxt('data/optos_fig6_sim.csv', delimiter=',')
sim_fig7 = np.loadtxt('data/optos_fig7_sim.csv', delimiter=',')
sim_fig8 = np.loadtxt('data/optos_fig8_sim.csv', delimiter=',')

# planar

struc = tmm_structure([Layer(si('200um'), Si)], Air, Air)
options.coherent = False
options.coherency_list = ['i']
RAT = tmm_structure.calculate(struc, options)

palhf = sns.color_palette("hls", 4)

fig = plt.figure()
plt.plot(sim_fig6[:,0], sim_fig6[:,1], '--', color=palhf[0], label= 'OPTOS - rear grating (a)')
plt.plot(wavelengths*1e9, RAT_fig6['A_bulk'][0], '-o', color=palhf[0], label='RayFlare - rear grating (a)', fillstyle='none')
plt.plot(sim_fig7[:,0], sim_fig7[:,1], '--', color=palhf[1],  label= 'OPTOS - front pyramids (b)')
plt.plot(wavelengths*1e9, RAT_fig7['A_bulk'][0], '-o', color=palhf[1],  label= 'RayFlare - front pyramids (b)', fillstyle='none')
plt.plot(sim_fig8[:,0], sim_fig8[:,1], '--', color=palhf[2],  label= 'OPTOS - grating + pyramids (c)')
plt.plot(wavelengths*1e9, RAT_fig8['A_bulk'][0], '-o', color=palhf[2],  label= 'RayFlare - grating + pyramids (c)', fillstyle='none')
plt.plot(wavelengths*1e9, RAT['A_per_layer'][:,0], '-k', label='Planar')
plt.plot(wavelengths*1e9, RAT_planar['A_bulk'][0], '--r')
plt.legend(loc='lower left')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorption in Si')
plt.xlim([900, 1200])
plt.ylim([0, 1])
plt.show()


theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])


path = get_savepath('default', options.project_name)
sprs = load_npz(os.path.join(path, SC_fig6[2].name + 'frontRT.npz'))

wl_to_plot = 1100e-9

wl_index = np.argmin(np.abs(wavelengths-wl_to_plot))

full = sprs[wl_index].todense()

summat = theta_summary(full, angle_vector, options['n_theta_bins'], 'front')

summat_r = summat[:options['n_theta_bins'], :]

summat_r = summat_r.rename({r'$\theta_{in}$': r'$\sin(\theta_{in})$', r'$\theta_{out}$': r'$\sin(\theta_{out})$'})

summat_r = summat_r.assign_coords({r'$\sin(\theta_{in})$': np.sin(summat_r.coords[r'$\sin(\theta_{in})$']).data,
                                    r'$\sin(\theta_{out})$': np.sin(summat_r.coords[r'$\sin(\theta_{out})$']).data})


palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)

fig = plt.figure()
ax = plt.subplot(111)
ax = summat_r.plot.imshow(ax=ax, cmap=seamap, vmax=0.3)
plt.show()

