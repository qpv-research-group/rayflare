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


nxy = 21

calc = False
options = {'wavelengths': np.array([1100e-9]), 'theta': 0 * np.pi / 180, 'phi': 0,
           'I_thresh': 0.02, 'nx': nxy, 'ny': nxy,
           'parallel': True, 'pol': 'u', 'n_rays': 2 * nxy ** 2, 'depth_spacing': si('1um'),
           'random_ray_position': False}


sizes = np.linspace(2, 5, 30)

res_rand = []


for sz in sizes:
    print('SIZE',sz)
    flat_surf = planar_surface(size=sz)
    triangle_surf = regular_pyramids(55, upright=True, size=sz)
    options['avoid_edges'] = False
    options['randomize'] = True
    rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si(200, 'um')], incidence=Air, transmission=Air)

    start = time()
    result_new1 = rtstr1.calculate(options)
    print(str(sz), time()-start)
    res_rand.append(result_new1)


A_rand = [np.asscalar(res_rand[i1]['A_per_layer'][0]) for i1 in range(len(sizes))]
R_rand = [np.asscalar(res_rand[i1]['R']) for i1 in range(len(sizes))]
T_rand = [np.asscalar(res_rand[i1]['T']) for i1 in range(len(sizes))]
R0_rand = [np.asscalar(res_rand[i1]['R0']) for i1 in range(len(sizes))]


res_reg = []

sizes_reg = np.linspace(2, 5, 100)
options['n_rays'] =  1*nxy**2
for sz in sizes_reg:
    print('SIZE',sz)
    flat_surf = planar_surface(size=sz)
    triangle_surf = regular_pyramids(55, upright=True, size=sz)
    options['avoid_edges'] = False
    options['randomize'] = False
    rtstr1 = rt_structure(textures=[triangle_surf, flat_surf],
                        materials = [Si],
                        widths=[si(200, 'um')], incidence=Air, transmission=Air)

    start = time()
    result_new1 = rtstr1.calculate(options)
    print(str(sz), time()-start)
    res_reg.append(result_new1)


A_reg = [np.asscalar(res_reg[i1]['A_per_layer'][0]) for i1 in range(len(sizes_reg))]
R_reg = [np.asscalar(res_reg[i1]['R']) for i1 in range(len(sizes_reg))]
T_reg = [np.asscalar(res_reg[i1]['T']) for i1 in range(len(sizes_reg))]
R0_reg = [np.asscalar(res_reg[i1]['R0']) for i1 in range(len(sizes_reg))]

plt.figure()
plt.plot(sizes_reg, A_reg, 'k-')
plt.plot(sizes_reg, R_reg, 'b-')
plt.plot(sizes_reg, T_reg, 'r-')
plt.plot(sizes_reg, R0_reg, 'g-')
plt.plot(sizes, A_rand, 'k--')
plt.plot(sizes, R_rand, 'b--')
plt.plot(sizes, T_rand, 'r--')
plt.plot(sizes, R0_rand, 'g--')
plt.legend(['A reg', 'R', 'T'])

plt.show()

plt.figure()
plt.hist(res_reg[0]['thetas'][0])
plt.show()

plt.figure()
plt.hist(res_reg[-1]['thetas'][0])
plt.show()