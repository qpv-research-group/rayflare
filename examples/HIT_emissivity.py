import numpy as np

from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options
from rayflare.angles import make_angle_vector

import matplotlib.pyplot as plt
import seaborn as sns

from cycler import cycler


import os

# new materials from data - uncomment to add to database

# from solcore.material_system import create_new_material
# cur_path = os.path.dirname(os.path.abspath(__file__))
# create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
# create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
# create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
# create_new_material('ITO_front', os.path.join(cur_path, 'data/front_ITO_n.txt'), os.path.join(cur_path, 'data/front_ITO_k.txt'))
# create_new_material('ITO_back', os.path.join(cur_path, 'data/back_ITO_n.txt'), os.path.join(cur_path, 'data/back_ITO_k.txt'))
# create_new_material('Si_UVtoMIR', os.path.join(cur_path, 'data/Si_IR_recon_n.txt'), os.path.join(cur_path, 'data/Si_IR_recon_k.txt'))
# create_new_material('Ag_Jiang', os.path.join(cur_path, 'data/Ag_UNSW_n.txt'), os.path.join(cur_path, 'data/Ag_UNSW_k.txt'))

# matrix multiplication
wavelengths = np.linspace(np.log(300), np.log(16*1000), 104)
wavelengths = np.round(np.floor(np.exp(wavelengths))*1e-9, 12)

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'HIT_notebook'
options.n_rays = 5000
options.n_theta_bins = 20
options.nx = 5
options.ny = 5
_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])
options.bulk_profile = True

Si = material('Si_UVtoMIR')()
Air = material('Air')()
ITO_front = material('ITO_front')()
ITO_back = material('ITO_back')()
Ag = material('Ag_Jiang')()
aSi_i = material('aSi_i')()
aSi_p = material('aSi_p')()
aSi_n = material('aSi_n')()


# stack based on doi:10.1038/s41563-018-0115-4
front_materials = [Layer(80e-9, ITO_front), Layer(6.5e-9, aSi_p), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_n), Layer(240e-9, ITO_back)]

# whether pyramids are upright or inverted is relative to front incidence.
# so if the same etch is applied to both sides of a slab of silicon, one surface
# will have 'upright' pyramids and the other side will have 'not upright' (inverted)
# pyramids in the model

surf = regular_pyramids(elevation_angle=55, upright=True)
surf_back = regular_pyramids(elevation_angle=55, upright=False)

front_surf = Interface('RT_TMM', texture = surf, layers=front_materials, name = 'HIT_front',
                       coherent=True)
back_surf = Interface('RT_TMM', texture = surf_back, layers=back_materials, name = 'HIT_back',
                      coherent=True)


bulk_Si = BulkLayer(170e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)

RAT = results[0]
results_per_pass = results[1]


R_per_pass = np.sum(results_per_pass['r'][0], 2)
R_0 = R_per_pass[0]
R_escape = np.sum(R_per_pass[1:, :], 0)

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)

results_per_layer_back = np.sum(results_per_pass['a'][1], 0)


allres = np.flip(np.stack((results_per_layer_front[:,0], results_per_layer_front[:,1]+results_per_layer_front[:,2],
                            RAT['A_bulk'][0, :],
                    results_per_layer_back[:,0] + results_per_layer_back[:,1], results_per_layer_back[:,2], RAT['T'][0, :])), 0)

# calculated photogenerated current (Jsc with 100% EQE)

spectr_flux = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                           output_units='photon_flux_per_m', concentration=1).spectrum(wavelengths)[1]

Jph_Si = q * np.trapz(RAT['A_bulk'][0] * spectr_flux, wavelengths)/10 # mA/cm2

pal = sns.cubehelix_palette(allres.shape[0], start=.5, rot=-.9)
pal.reverse()
cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)


from scipy.ndimage.filters import gaussian_filter1d
#
# ysmoothed = gaussian_filter1d(allres, sigma=2, axis=0)
#
# bulk_A_text= ysmoothed[:,4]

emissivity = np.loadtxt('data/emissivity.csv', delimiter=',')
noITO_emissivity = np.loadtxt('data/emissivity_noITO.csv', delimiter=',')

# plot total R, A, T
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
ax.semilogx(options['wavelengths']*1e6, R_escape + R_0, '--k', label=r'$R_{total}$')
ax.semilogx(options['wavelengths']*1e6, R_0, '-.k', label=r'$R_0$')
ax.stackplot(options['wavelengths']*1e6, allres, labels=['Ag', 'Back ITO', 'a-Si (back)', 'Bulk Si', 'a-Si (front)', 'Front ITO'])
ax.semilogx(emissivity[:,0], emissivity[:,1], '-k')
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel('Absorption/Emissivity')
ax.set_xlim(min(options['wavelengths']*1e6), max(options['wavelengths']*1e6))
ax.set_ylim(0, 1)
plt.show()

