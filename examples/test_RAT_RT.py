from transfer_matrix_method.transfer_matrix import TMM
from solcore.structure import Layer
from solcore import material
from angles import make_angle_vector, theta_summary

from options import default_options

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl


wavelengths = np.linspace(300, 1200, 20)*1e-9

options = default_options
options['n_theta_bins'] = 10
options['wavelengths'] = wavelengths
options['c_azimuth'] = 0.5
options['phi_symmetry'] = 2*np.pi


options.project_name = 'testing'

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                       options['c_azimuth'])


GaAs = material('GaAs')()
Ge = material('Ge')()
Si = material('Glass_Rubin')()

Air = material('Air')()

layers  = [Layer(500e-9, GaAs), Layer(1000e-9, Ge)]

theta_all = np.unique(angle_vector[:,1])
theta_r = theta_all[:options['n_theta_bins']]
theta_t = theta_all[options['n_theta_bins']:]


problem = TMM(layers, Air, Si, 'test', options, True, None, [], 'front', False)

full = problem[0].todense()[19]



summat, Rsum, Tsum = theta_summary(full, angle_vector)

summat = xr.DataArray(summat, dims=[r'$\theta_{out}$', r'$\theta_{in}$'], coords={r'$\theta_{out}$': theta_all, r'$\theta_{in}$': theta_all})

# Rth = summat[0:options['n_theta_bins'],:]
# Tth = summat[options['n_theta_bins']:, :]
#
# theta_r = np.unique(angle_vector[:,1])[:options['n_theta_bins']]
# theta_t = np.unique(angle_vector[:,1])[options['n_theta_bins']:]
# Rth = xr.DataArray(Rth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'])#, coords={r'$\sin(\theta_{out})$': np.linspace(0,1,options['n_theta_bins']),
#                                                                            # r'$\sin(\theta_{in})$': np.linspace(0,1,options['n_theta_bins'])})
# Tth = xr.DataArray(Tth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'])#, coords={r'$\sin(\theta_{out})$': np.linspace(1,0,options['n_theta_bins']),
#                                                                             #r'$\sin(\theta_{in})$': np.linspace(0,1,options['n_theta_bins'])})
#
# Rth = xr.DataArray(Rth, dims=[r'$\theta_{out}$', r'$\theta_{in}$'])#, coords={r'$\theta_{out}$': theta_r,
#                                                                            # r'$\theta_{in}$': theta_r})
# Tth = xr.DataArray(Tth, dims=[r'$\theta_{out}$', r'$\theta_{in}$'])#, coords={r'$\theta_{out}$': theta_t,
#                                                                            # r'$\theta_{in}$': theta_r})


palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
palhf.reverse()
seamap = mpl.colors.ListedColormap(palhf)

fig = plt.figure()
ax = plt.subplot(111)
ax = summat.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('matrix.png', bbox_inches='tight', format='png')
#ax = Tth.plot.imshow(ax=ax)

plt.show()
#


problem_back = TMM(layers, Air, Si, 'test', options, True, None, [], 'rear', False)

full_back = problem_back[0].todense()[19]



summat_back, Rsum, Tsum = theta_summary(full_back, angle_vector)


summat_back = xr.DataArray(summat_back, dims=[r'$\theta_{out}$', r'$\theta_{in}$'], coords={r'$\theta_{out}$': theta_all, r'$\theta_{in}$': theta_all})
# Rth = summat[0:options['n_theta_bins'],:]
# Tth = summat[options['n_theta_bins']:, :]
#
# theta_r = np.unique(angle_vector[:,1])[:options['n_theta_bins']]
# theta_t = np.unique(angle_vector[:,1])[options['n_theta_bins']:]
# Rth = xr.DataArray(Rth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'])#, coords={r'$\sin(\theta_{out})$': np.linspace(0,1,options['n_theta_bins']),
#                                                                             #r'$\sin(\theta_{in})$': np.linspace(0,1,options['n_theta_bins'])})
# Tth = xr.DataArray(Tth, dims=[r'$\sin(\theta_{out})$', r'$\sin(\theta_{in})$'])#, coords={r'$\sin(\theta_{out})$': np.linspace(1,0,options['n_theta_bins']),
#                                                                             #r'$\sin(\theta_{in})$': np.linspace(0,1,options['n_theta_bins'])})
#
# Rth = xr.DataArray(Rth, dims=[r'$\theta_{out}$', r'$\theta_{in}$'])#, coords={r'$\theta_{out}$': theta_r,
#                                                                     #        r'$\theta_{in}$': theta_r})
# Tth = xr.DataArray(Tth, dims=[r'$\theta_{out}$', r'$\theta_{in}$'])#, coords={r'$\theta_{out}$': theta_t,
#                                                                     #        r'$\theta_{in}$': theta_r})


fig = plt.figure()
ax = plt.subplot(111)
ax = summat_back.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('matrix.png', bbox_inches='tight', format='png')
#ax = Tth.plot.imshow(ax=ax)

plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = Tth.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()

whole_mat = xr.concat((summat[:, :options['n_theta_bins']], summat_back[:, options['n_theta_bins']:]), dim=r'$\theta_{in}$')

whole_mat_imshow = whole_mat.rename({r'$\theta_{in}$': 'theta_in', r'$\theta_{out}$': 'theta_out'})

whole_mat_imshow = whole_mat_imshow.interp(theta_in = np.linspace(0, np.pi, 200), theta_out =  np.linspace(0, np.pi, 200))

whole_mat_imshow = whole_mat_imshow.rename({'theta_in': r'$\theta_{in}$', 'theta_out' : r'$\theta_{out}$'})

fig = plt.figure()
ax = plt.subplot(111)
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('matrix.png', bbox_inches='tight', format='png')
#ax = Tth.plot.imshow(ax=ax)

plt.show()