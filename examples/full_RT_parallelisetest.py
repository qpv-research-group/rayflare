from rayflare.ray_tracing.rt import rt_structure
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
Si = material('Si')()
GaAs = material('GaAs')()
Ge = material('Ge')()


flat_surf = planar_surface()
triangle_surf = regular_pyramids(10)

#options = {'wavelengths': np.linspace(700, 1700, 100)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
#           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
#           'parallel': True, 'pol': 'p', 'n_rays': 2000, 'depth_spacing': 1, 'n_jobs': -1}
options = {'wavelengths': np.linspace(700, 1700, 7)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
           'parallel': True, 'pol': 'p', 'n_rays': 2000, 'depth_spacing': si('1um'), 'n_jobs': -1}
#structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [GaAs, Si, Ge],
#                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

rtstr = rt_structure(textures=[flat_surf, flat_surf, flat_surf, flat_surf],
                    materials = [GaAs, Si, Ge],
                    widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
start = time()
result_new = rtstr.calculate(options)
print(time()-start)
result = result_new

plt.figure()
plt.plot(options['wavelengths']*1e9, result['R'])
plt.plot(options['wavelengths']*1e9, result['T'])
plt.plot(options['wavelengths']*1e9, result['A_per_layer'])
#plt.ylim(0,1)
plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.show()

plt.figure()
plt.plot(result['profile'].T)
plt.show()




options = {'wavelengths': np.linspace(700, 1700, 7)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
           'parallel': False, 'pol': 'p', 'n_rays': 2000, 'depth_spacing': si('1um'), 'n_jobs': -1}
#structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [GaAs, Si, Ge],
#                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                    materials = [GaAs, Si, Ge],
                    widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)

start = time()
result_old = rtstr.calculate(options)
print(time()-start)
result = result_old

plt.figure()
plt.plot(options['wavelengths']*1e9, result['R'])
plt.plot(options['wavelengths']*1e9, result['T'])
plt.plot(options['wavelengths']*1e9, result['A_per_layer'])
#plt.ylim(0,1)
plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.show()

plt.figure()
plt.plot(result['profile'].T)
plt.show()


from transfer_matrix_method.tmm import tmm_structure
from solcore.structure import Layer

stack = [Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)]

strt = tmm_structure(stack, coherent=False, coherency_list=['i', 'i', 'i'],
                     no_back_reflection=False)

output = strt.calculate(options['wavelengths']*1e9, angle=options['theta'], pol=options['pol'],
                        profile=True, nm_spacing=1000, layers=[1,2,3])

plt.figure()
plt.plot(options['wavelengths']*1e9, output['R'])
plt.plot(options['wavelengths']*1e9, output['T'])
plt.plot(options['wavelengths']*1e9, output['A_per_layer'].T)
plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.ylim(0,1)
plt.show()

plt.figure()
plt.plot(output['profile'].T)
plt.show()
