from ray_tracing.rt_lookup import RT
from textures.standard_rt_textures import regular_pyramids
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from structure import RTgroup

from solcore.structure import Layer
from solcore import material
from angles import make_angle_vector, theta_summary

from options import default_options

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

n_wl = 8*5

wavelengths = np.linspace(300, 1800, n_wl)*1e-9

options = default_options
options['n_theta_bins'] = 100
options['wavelengths'] = wavelengths
options['c_azimuth'] = 0.1
options['phi_symmetry'] = np.pi/2
options['n_rays'] = 200000
options['parallel'] = True


options.project_name = 'testing'

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                       options['c_azimuth'])

surf = regular_pyramids(elevation_angle=0, upright=True) # [texture, reverse]

group = RTgroup(textures=[surf])


GaAs = material('GaAs')()
Ge = material('Ge')()
Si = material('Si')()

Air = material('Air')()

#results_front = RT(group, Air, Si, 'test', options, 0, 'front', 0, False)

# lookup table:
layers  = [Layer(500e-9, GaAs), Layer(200e-9, Ge)]

make_TMM_lookuptable(layers, Air, Si, 'test', options)

results_front = RT(group, Air, Si, 'test', options, 1, 'front', len(layers))

full = results_front[0].todense()[n_wl-1]

theta_all = np.unique(angle_vector[:,1])
theta_r = theta_all[:options['n_theta_bins']]
theta_t = theta_all[options['n_theta_bins']:]


summat = theta_summary(full, angle_vector, options['n_theta_bins'])

Rth = summat[0:options['n_theta_bins'],:]
Tth = summat[options['n_theta_bins']:, :]
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
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = summat.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()
#
# palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
# palhf.reverse()
# seamap = mpl.colors.ListedColormap(palhf)
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = Rth.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = Tth.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()
#


#results_back= RT(group, Air, Si, 'test', options, 0, 'rear', 0, False)

results_back = RT(group, Air, Si, 'test', options, 1, 'rear', len(layers))


full = results_back[0].todense()[n_wl-1]

theta_all = np.unique(angle_vector[:,1])
theta_r = theta_all[:options['n_theta_bins']]
theta_t = theta_all[options['n_theta_bins']:]


summat_back = theta_summary(full, angle_vector, options['n_theta_bins'], "rear")


Rth = summat_back[options['n_theta_bins']:, :]
Tth = summat_back[0:options['n_theta_bins']:, :]
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
#
#
# palhf = sns.cubehelix_palette(256, start=.5, rot=-.9)
# palhf.reverse()
# seamap = mpl.colors.ListedColormap(palhf)
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = summat_back.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = Rth.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()
#
# fig = plt.figure()
# ax = plt.subplot(111)
# ax = Tth.plot.imshow(ax=ax, cmap=seamap)
# #ax = plt.subplot(212)
# #fig.savefig('matrix.png', bbox_inches='tight', format='png')
# #ax = Tth.plot.imshow(ax=ax)
#
# plt.show()




whole_mat = xr.concat((summat, summat_back), dim=r'$\theta_{in}$')

whole_mat_imshow = whole_mat.rename({r'$\theta_{in}$': 'theta_in', r'$\theta_{out}$': 'theta_out'})

whole_mat_imshow = whole_mat_imshow.interp(theta_in = np.linspace(0, np.pi, 100), theta_out =  np.linspace(0, np.pi, 100))

whole_mat_imshow = whole_mat_imshow.rename({'theta_in': r'$\theta_{in}$', 'theta_out' : r'$\theta_{out}$'})


fig = plt.figure()
ax = plt.subplot(111)
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)

#ax = Tth.plot.imshow(ax=ax)

plt.show()

from sparse import nansum

Rf = nansum(results_front[0][:, 0:(int(len(angle_vector)/2))], 1)[:, 10].todense()
Tf = nansum(results_front[0][:, (int(len(angle_vector)/2)):], 1)[:, 10].todense()
Af_1 = results_front[1][:,0,10].todense()
Af_2 = results_front[1][:,1,10].todense()


Rb = nansum(results_back[0][:, (int(len(angle_vector)/2)):], 1)[:, -11].todense()
Tb = nansum(results_back[0][:, 0:(int(len(angle_vector)/2))], 1)[:, -11].todense()
Ab_1 = results_back[1][:,0,-11].todense()
Ab_2 = results_back[1][:,1,-11].todense()


plt.figure()
plt.plot(wavelengths*1e9, Rf, label='Rf')
plt.plot(wavelengths*1e9, Tf, label='Tf')
plt.plot(wavelengths*1e9, Af_1, '--', label='Af_1')
plt.plot(wavelengths*1e9, Af_2, ':', label='Af_2')
plt.plot(wavelengths*1e9, Rb, label='Rb')
plt.plot(wavelengths*1e9, Tb, label='Tb')
plt.plot(wavelengths*1e9, Ab_1,  '--', label='Ab_1')
plt.plot(wavelengths*1e9, Ab_2,  ':',label='Ab_2')
plt.plot(wavelengths*1e9, Rf+Tf+Af_1+Af_2, 'k-')
plt.plot(wavelengths*1e9, Rb+Tb+Ab_1+Ab_2, 'r-')
plt.legend()
plt.show()
