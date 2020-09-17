import numpy as np
import os
# solcore imports
from solcore.structure import Layer
from solcore import material
from solcore import si
from matrix_formalism.process_structure import process_structure
from matrix_formalism.multiply_matrices import calculate_RAT
from angles import theta_summary
from textures.standard_rt_textures import regular_pyramids
from solcore.material_system import create_new_material
import matplotlib.pyplot as plt
from sparse import load_npz

from config import results_path

from angles import make_angle_vector
from config import results_path
from sparse import load_npz



angle_degrees_in = 0
# matrix multiplication
wavelengths = np.linspace(900, 1200, 30)*1e-9
options = {'nm_spacing': 0.5,
           'project_name': 'optos_checks_2',
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
           'n_rays': 500000,
           'random_angles': False,
           'nx': 15, 'ny': 15,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           'only_incidence_angle': True
           }

Si = material('Si_OPTOS')()

sprs = load_npz(os.path.join(results_path, options['project_name'], 'planar_back' + str(options['n_rays']) + 'frontRT.npz'))


_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])

R = sprs.todense()[:, 0:int(len(angle_vector)/2), :]

a, unique_index = np.unique(angle_vector[:,1], return_index=True)
unique_index = unique_index[unique_index < 1300]

only_one_theta = R[:,unique_index, unique_index]
plt.figure()
a = plt.imshow(only_one_theta, extent=[0,1, 900, 1200], aspect='auto')
plt.colorbar(a)
plt.show()


from rayflare.ray_tracing.rt import calc_R
calc_R_result = np.zeros((len(options['wavelengths']), len(unique_index)))

for i1, angle in enumerate(unique_index):

    calc_R_result[:, i1] = calc_R(Si.n(options['wavelengths']), 1, angle_vector[angle, 1], 'u')


calc_R_result[np.where(np.isnan(calc_R_result))] = 1
plt.figure()
a = plt.imshow(calc_R_result, extent=[0,1, 900, 1200], aspect='auto')
plt.colorbar(a)
plt.show()