from solcore import material
from solcore import si

from structure import Structure, Layer, Texture, Surface, RTgroup
from ray_tracing.rt_lookup import RTSurface, RT, single_ray, overall_bin, RT_wl
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
wavelengths = np.linspace(700, 1100, 4)*1e-9
#pool = Pool(processes = 4)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4, 'theta': 0, 'phi': 0,
            'nx': 2, 'ny': 2, 'max_passes': 50, 'parallel': True, 'n_rays': 100000,
            'phi_symmetry': np.pi/2, 'n_theta_bins': 100, 'c_azimuth': 0.25,
            'random_angles': False, 'pol': 's', 'struct_name': 'testing', 'Fr_or_TMM': 1}#,
            #'pool': pool}


surf = Surface('RT', None, texture = surf, depth_spacing = si('1nm'))

start = time()
group = RTgroup(textures=[surf.texture])

incidence = Air
transmission = Si

allArrays, absArrays = RT(group, incidence, transmission, 'GaAsGaAsstack', 2, options, 'front')
print('Time taken = ', time() - start, ' s')

allArrays, absArrays = RT(group, transmission, incidence, 'GaAsGaAsstack', 2, options, 'rear')
print('Time taken = ', time() - start, ' s')

from sparse import load_npz
allArrays = load_npz('C:\\Users\\pmpea\\Box Sync\\Optics package\\results\\testing\\GeGaAsstackfrontRT.npz')

out_mat = allArrays[0]
outfull = out_mat.todense()
#absArrays = absArrays.todense()

from angles import theta_summary, theta_summary_A, make_angle_vector

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'],
                                                       options['phi_symmetry'],
                                                       options['c_azimuth'])

theta_sum, R, T = theta_summary(outfull, angle_vector)

plt.figure()
plt.imshow(theta_sum, cmap='hot', interpolation='nearest')
plt.show()

A_sum = theta_summary_A(absArrays[0], angle_vector)
thetas = (theta_intv[1:]+theta_intv[:-1])/2
thetas = thetas[:int(len(thetas)/2)]
plt.figure()
plt.plot(thetas, A_sum[0])
plt.plot(thetas, A_sum[1])


