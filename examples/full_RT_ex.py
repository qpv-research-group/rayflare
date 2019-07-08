from ray_tracing.rt_lookup import full_RT
from structure import RTgroup
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np

Air = material('Air')()
Si = material('Si')()
GaAs = material('GaAs')()
Ge = material('Ge')()


flat_surf = planar_surface()
triangle_surf = regular_pyramids(10)

options = {'wavelengths': np.linspace(700, 1700, 10)*1e-9, 'theta': 45*np.pi/180, 'phi': 0,
           'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
           'parallel': False, 'pol': 'p', 'n_rays': 1000}
#structure = RTgroup(textures=[flat_surf, flat_surf, flat_surf, flat_surf], materials = [GaAs, Si, Ge],
#                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

structure = RTgroup(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf], materials = [GaAs, Si, Ge],
                    widths=[si('100um'), si('70um'), si('50um')], depth_spacing=1)

result = full_RT(structure, Air, Air, options)
#result = full_RT(structure, Air, Air, options)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(options['wavelengths']*1e9, result[0])
plt.plot(options['wavelengths']*1e9, result[1])
plt.plot(options['wavelengths']*1e9, result[2])
plt.ylim(0,1)
plt.plot(options['wavelengths']*1e9, result[0]+result[1]+np.sum(result[2],1), 'g')
#plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
#plt.show()

#options = {'wavelengths': np.linspace(700, 1700, 10)*1e-9, 'theta': 0*np.pi/180, 'phi': 0, 'I_thresh': 1e-4, 'nx': 5, 'ny': 5,
#           'parallel': False, 'pol': 's', 'n_rays': 1000}

#result = full_RT(structure, Air, Air, options)
#plt.plot(options['wavelengths']*1e9, result[0], 'r--')
#plt.plot(options['wavelengths']*1e9, result[1], 'b--')
#plt.plot(options['wavelengths']*1e9, result[2], 'g--')
plt.show()
# incoherent TMM comparison

from transfer_matrix_method.transfer_matrix import OptiStack, calculate_rat
from solcore.structure import Layer

stack = OptiStack([Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)], False, Air, Air)

output = calculate_rat(stack, options['wavelengths']*1e9, angle=options['theta'], pol=options['pol'],
                  coherent=False, coherency_list=['i', 'i', 'i'], profile=False)

plt.figure()
plt.plot(options['wavelengths']*1e9, output['R'])
plt.plot(options['wavelengths']*1e9, output['T'])
plt.plot(options['wavelengths']*1e9, output['A_per_layer'].T)
plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.ylim(0,1)
plt.show()
