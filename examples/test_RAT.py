from transfer_matrix_method.tmm import TMM
from solcore.structure import Layer
from solcore import material
from angles import make_angle_vector, theta_summary

from options import default_options

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

n_wl = 80

wavelengths = np.linspace(300, 1800, n_wl)*1e-9

options = default_options
options['n_theta_bins'] = 100
options['wavelengths'] = wavelengths
options['c_azimuth'] = 0.1
options['phi_symmetry'] = np.pi/2
options['parallel'] = True
options['pol'] = 'u'



options.project_name = 'testing'

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                       options['c_azimuth'])


GaAs = material('GaAs')()
Ge = material('Ge')()
Si = material('Si')()

Air = material('Air')()

layers  = [Layer(500e-9, GaAs), Layer(200e-9, Ge)]

theta_all = np.unique(angle_vector[:,1])
theta_r = theta_all[:options['n_theta_bins']]
theta_t = theta_all[options['n_theta_bins']:]


results_front = TMM(layers, Air, Si, 'test', options, True, None, [], 'front', False)

full = results_front[0].todense()[n_wl-1]

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


results_back = TMM(layers, Air, Si, 'test', options, True, None, [], 'rear', False)

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

angle_index = 100

from sparse import nansum

Rf = nansum(results_front[0][:, 0:(int(len(angle_vector)/2))], 1)[:, angle_index ].todense()
Tf = nansum(results_front[0][:, (int(len(angle_vector)/2)):], 1)[:, angle_index ].todense()
Af_1 = results_front[1][:,0,angle_index].todense()
Af_2 = results_front[1][:,1,angle_index].todense()


Rb = nansum(results_back[0][:, (int(len(angle_vector)/2)):], 1)[:, -(angle_index +1)].todense()
Tb = nansum(results_back[0][:, 0:(int(len(angle_vector)/2))], 1)[:, -(angle_index +1)].todense()
Ab_1 = results_back[1][:,0,-(angle_index +1)].todense()
Ab_2 = results_back[1][:,1,-(angle_index +1)].todense()


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

#  compare with just TMM
inc_angle = angle_vector[angle_index, 1]*180/np.pi
from solcore.absorption_calculator import OptiStack, calculate_rat

layers  = [Layer(500e-9, GaAs), Layer(200e-9, Ge)]
OptSt = OptiStack(layers, no_back_reflection=False, substrate=Si, incidence=Air)

RAT = calculate_rat(OptSt, options['wavelengths']*1e9, angle=inc_angle, no_back_reflection=False, pol=options['pol'])

OptSt_flip = OptiStack(layers[::-1], no_back_reflection=False, substrate=Air, incidence=Si)

RAT_flip = calculate_rat(OptSt_flip, options['wavelengths']*1e9, angle=inc_angle, no_back_reflection=False, pol=options['pol'])

plt.figure()
plt.plot(wavelengths*1e9, RAT['R'], 'r-')
plt.plot(wavelengths*1e9, RAT['T'], 'r--')
plt.plot(wavelengths*1e9, Rf, label='Rf')
plt.plot(wavelengths*1e9, Tf, label='Tf')
plt.plot(wavelengths*1e9, RAT['A_per_layer'][1], label='A1_TMM')
plt.plot(wavelengths*1e9, RAT['A_per_layer'][2], label='A2_TMM')
plt.plot(wavelengths*1e9, Af_1, label='Af1')
plt.plot(wavelengths*1e9, Af_2, label='Af2')
plt.plot(wavelengths*1e9, RAT['R'] + RAT['T'] + RAT['A_per_layer'][1] + RAT['A_per_layer'][2], 'k-')
# plt.plot(wavelengths*1e9, RAT_flip['R'], 'k-')
# plt.plot(wavelengths*1e9, RAT_flip['T'], 'k--')
# plt.plot(wavelengths*1e9, RAT_flip['A_per_layer'][1] + RAT_flip['A_per_layer'][2])
# plt.plot(wavelengths*1e9, RAT_flip['R'] + RAT_flip['T'] + RAT_flip['A_per_layer'][1] + RAT_flip['A_per_layer'][2], 'k-')
plt.legend()
plt.show()
