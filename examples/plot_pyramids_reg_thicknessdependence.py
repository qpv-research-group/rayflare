from textures.standard_rt_textures import regular_pyramids, random_pyramids
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl



from ray_tracing.rt import rt_structure
from time import time
from textures.standard_rt_textures import regular_pyramids, planar_surface
from solcore import material
from solcore import si
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
pal = sns.cubehelix_palette()

cols = cycler('color', pal)

params = {'legend.fontsize': 'small',
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize': 'small',
          'ytick.labelsize': 'small',
          'axes.prop_cycle': cols}

plt.rcParams.update(params)

Air = material('Air')()
Si = material('566', nk_db=True)()

size = 2
nxy = 21

calc = False
options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 1 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False}


thickness = np.linspace(190, 210, 100)
res_reg = []
res_rand = []

for th in thickness:


    flat_surf = planar_surface()
    triangle_surf = regular_pyramids(55, upright=True, size=size)
    options['avoid_edges'] = False
    options['randomize'] = False
    rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si(th, 'um')], incidence=Air, transmission=Air)

    start = time()
    result_new1 = rtstr1.calculate(options)
    print(str(size), time()-start)
    res_reg.append(result_new1['A_per_layer'][:,0][0])

for th in thickness:
    flat_surf = planar_surface()
    triangle_surf = regular_pyramids(55, upright=True, size=size)
    options['avoid_edges'] = False
    options['randomize'] = True
    rtstr2 = rt_structure(textures=[triangle_surf, flat_surf],
                         materials=[Si],
                         widths=[si(th, 'um')], incidence=Air, transmission=Air)

    start = time()
    result_new2 = rtstr2.calculate(options)
    print(str(size), time()-start)
    res_rand.append(result_new2['A_per_layer'][:,0][0])


plt.figure()
plt.plot(thickness, res_rand, label='rand')
plt.plot(thickness, res_reg, label='reg')
plt.legend()
plt.show()