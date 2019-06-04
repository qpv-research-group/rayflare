from solcore import material
from solcore import si

from structure import Structure, BulkLayer, Texture, Interface, RTgroup
from ray_tracing.rt_lookup import RTSurface, RT, overall_bin, RT_wl
import numpy as np
import math
from time import time

import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool

Si = material('Si')()
Air = material('Air')()

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

x = np.array([0, 0, Lx, Lx])
y = np.array([0, Ly, Ly, 0])
z = np.array([0, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)
wavelengths = np.linspace(600, 1100, 4)*1e-9
#pool = Pool(processes = 4)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4, 'theta': 0, 'phi': 0,
            'nx': 2, 'ny': 2, 'max_passes': 50, 'parallel': True, 'n_jobs': -1, 'n_rays': 400000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': False, 'pol': 's', 'struct_name': 'testing', 'Fr_or_TMM': 1,
            'surf_name': 'GaInPGaAsonSi', 'n_absorbing_layers': 2, 'front_or_rear': 'front',
            'prof_indices': [0,1]}#,
            #'pool': pool}


surf = Interface('RT', None, texture = surf, depth_spacing = si('1nm'))

start = time()
group = RTgroup(textures=[surf.texture])

incidence = Air
transmission = Si

allArrays, absArrays, prof_data = RT(group, incidence, transmission, options)
print('Time taken = ', time() - start, ' s')

options['front_or_rear'] = 'rear'

allArrays, absArrays, prof_data = RT(group, transmission, incidence, options)
print('Time taken = ', time() - start, ' s')
#
# out_mat = allArrays[0]
# outfull = out_mat.todense()
# absArrays = absArrays.todense()
# #
# from angles import theta_summary, theta_summary_A, make_angle_vector
# #
# theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'],
#                                                        options['phi_symmetry'],
#                                                        options['c_azimuth'])
#
#
# theta_sum, R, T = theta_summary(outfull, angle_vector)
#
# plt.figure()
# plt.imshow(theta_sum, cmap='hot', interpolation='nearest')
# plt.show()
#
# A_sum = theta_summary_A(absArrays[0], angle_vector)
# thetas = (theta_intv[1:]+theta_intv[:-1])/2
# thetas = thetas[:int(len(thetas)/2)]
# plt.figure()
# plt.plot(thetas, A_sum[0])
# plt.plot(thetas, A_sum[1])
#
# import xarray as xr
