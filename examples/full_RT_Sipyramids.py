from ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

pal = sns.cubehelix_palette(7, start=.5, rot=-.9)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

Air = material('Air')()
Si = material('Si_OPTOS')()


flat_surf = planar_surface()
triangle_surf = regular_pyramids(55, upright=False)

#options = {'wavelengths': np.linspace(700, 1700, 100)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
#           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
#           'parallel': True, 'pol': 'p', 'n_rays': 2000, 'depth_spacing': 1, 'n_jobs': -1}

options = {'wavelengths': np.linspace(1050, 1200, 20)*1e-9, 'theta': 0*np.pi/180, 'phi': 0,
           'I_thresh': 1e-3, 'nx': 20, 'ny': 20,
           'parallel': False, 'pol': 'u', 'n_rays': 2000, 'depth_spacing': si('1um'), 'n_jobs': -1}
#structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [GaAs, Si, Ge],
#                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

rtstr = rt_structure(textures=[triangle_surf, flat_surf],
                    materials = [Si],
                    widths=[si('200um')], incidence=Air, transmission=Air)
start = time()
result_new = rtstr.calculate(options)
print(time()-start)
result = result_new


PVlighthouse = np.loadtxt('data/RAT_data.csv', delimiter=',', skiprows=1)
#meas = np.loadtxt('data/optos_fig7_data.csv', delimiter=',')
plt.figure()
plt.plot(options['wavelengths']*1e9, result['R'], 'k-')
plt.plot(options['wavelengths']*1e9, result['T'], 'r-')
plt.plot(options['wavelengths']*1e9, result['A_per_layer'], 'g-')
plt.plot(PVlighthouse[:,0], PVlighthouse[:,2], 'k--')
plt.plot(PVlighthouse[:,0], PVlighthouse[:,9], 'r--')
plt.plot(PVlighthouse[:,0], PVlighthouse[:,5], 'g--')
#plt.ylim(0,1)
plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
#plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.show()

plt.figure()
plt.plot(result['profile'][21].T)
plt.show()


