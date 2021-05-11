from rayflare.ray_tracing import rt_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from rayflare.transfer_matrix_method import tmm_structure
from solcore.structure import Layer

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
triangle_surf = regular_pyramids(30)

options = default_options()

options.wavelengths = np.linspace(700, 1800, 7)*1e-9
options.theta_in = 45*np.pi/180
options.nx = 5
options.ny = 5
options.pol = 'p'
options.n_rays = 2000
options.depth_spacing = 1e-6


rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                    materials = [GaAs, Si, Ge],
                    widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)
result_new = rtstr.calculate(options)
result = result_new

plt.figure()
plt.plot(options['wavelengths']*1e9, result['R'])
plt.plot(options['wavelengths']*1e9, result['T'])
plt.plot(options['wavelengths']*1e9, result['A_per_layer'])
plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
plt.plot(options['wavelengths']*1e9, result['R0'], '--')
plt.legend(['R', 'T', 'L1', 'L2', 'L3', 'tot', 'R0'])
plt.show()

plt.figure()
plt.plot(result['profile'].T)
plt.show()


options.parallel = False


rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf, triangle_surf],
                    materials = [GaAs, Si, Ge],
                    widths=[si('100um'), si('70um'), si('50um')], incidence=Air, transmission=Air)

result_old = rtstr.calculate(options)
result = result_old

plt.figure()
plt.plot(options['wavelengths']*1e9, result['R'])
plt.plot(options['wavelengths']*1e9, result['T'])
plt.plot(options['wavelengths']*1e9, result['A_per_layer'])
plt.plot(options['wavelengths']*1e9, result['R']+result['T']+np.sum(result['A_per_layer'],1), 'g')
plt.plot(options['wavelengths']*1e9, result['R0'])
plt.legend(['R', 'T', 'L1', 'L2', 'L3', 'tot', 'R0'])
plt.show()

plt.figure()
plt.plot(result['profile'].T)
plt.show()


stack = [Layer(si('100um'), GaAs), Layer(si('70um'), Si), Layer(si('50um'), Ge)]

strt = tmm_structure(stack, incidence=Air, transmission=Air, no_back_reflection=False)

output = strt.calculate(options, profile=True, layers=[1,2,3])

plt.figure()
plt.plot(options['wavelengths']*1e9, output['R'])
plt.plot(options['wavelengths']*1e9, output['T'])
plt.plot(options['wavelengths']*1e9, output['A_per_layer'])
plt.legend(['R', 'T', 'L1', 'L2', 'L3'])
plt.ylim(0,1)
plt.show()

plt.figure()
plt.plot(output['profile'].T)
plt.show()
