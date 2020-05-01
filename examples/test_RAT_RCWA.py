from ray_tracing.rt_lookup import RT
from rigorous_coupled_wave_analysis.rcwa import rcwa_structure, RCWA
from textures.standard_rt_textures import regular_pyramids
from transfer_matrix_method.lookup_table import make_TMM_lookuptable
from structure import RTgroup

from solcore.structure import Layer
from solcore import material
from angles import make_angle_vector, theta_summary
from solcore.absorption_calculator import OptiStack

from options import default_options

import numpy as np
import seaborn as sns
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

n_wl = 80
orders = 4
wavelengths = np.linspace(300, 1800, n_wl)*1e-9

options = default_options
options['n_theta_bins'] = 20
options['wavelengths'] = wavelengths
options['c_azimuth'] = 0.1
options['phi_symmetry'] = np.pi/2
options['parallel'] = True
options['pol'] = 'u'


options.project_name = 'testing'

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                       options['c_azimuth'])

surf = regular_pyramids(elevation_angle=0, upright=True) # [texture, reverse]

GaAs = material('GaAs')()
Ge = material('Ge')()
Si = material('Si')()

Air = material('Air')()

layers  = [Layer(500e-9, GaAs), Layer(200e-9, Ge)]
# optist = OptiStack(layers, False, Si, Air)
# lookup table:
size = ((500,0), (0,500))

results_front = RCWA(layers, size, orders, options, Air, Si, only_incidence_angle=False, front_or_rear='front',
                     surf_name='testRCWA', detail_layer=False, save=True)

#str = rcwa_structure(layers, size, 4, options, Air, Si)


full = results_front[0].todense()[n_wl-1]

summat = theta_summary(full, angle_vector, options['n_theta_bins'])

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
plt.show()


results_back = RCWA(layers, size, orders, options, Air, Si, only_incidence_angle=False, front_or_rear='rear',
                    surf_name='testRCWA', detail_layer=False, save=True)

#str = rcwa_structure(layers, size, 4, options, Air, Si)


full = results_back[0].todense()[n_wl-1]

summat_back = theta_summary(full, angle_vector, options['n_theta_bins'], front_or_rear="rear")

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
ax = summat_back.plot.imshow(ax=ax, cmap=seamap)
plt.show()




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

angle_index = 400

from sparse import nansum

Rf = nansum(results_front[0][:, 0:(int(len(angle_vector)/2))], 1)[:, angle_index].todense()
Tf = nansum(results_front[0][:, (int(len(angle_vector)/2)):], 1)[:, angle_index].todense()
Af_1 = results_front[1][:,0,angle_index].todense()
Af_2 = results_front[1][:,1,angle_index].todense()


Rb = nansum(results_back[0][:, (int(len(angle_vector)/2)):], 1)[:, -(angle_index+1)].todense()
Tb = nansum(results_back[0][:, 0:(int(len(angle_vector)/2))], 1)[:, -(angle_index+1)].todense()
Ab_1 = results_back[1][:,0,-(angle_index+1)].todense()
Ab_2 = results_back[1][:,1,-(angle_index+1)].todense()


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


plt.figure()
plt.plot(wavelengths*1e9, results_front[2][:, angle_index], 'r-')
plt.plot(wavelengths*1e9, results_front[3][:, angle_index], 'r--')
plt.plot(wavelengths*1e9, results_back[2][:, -(angle_index+1)], 'k-')
plt.plot(wavelengths*1e9, results_back[3][:, -(angle_index+1)], 'k--')
plt.show()

inc_angle = angle_vector[angle_index, 1]*180/np.pi

from solcore.absorption_calculator import OptiStack, calculate_rat

layers  = [Layer(500e-9, GaAs), Layer(200e-9, Ge)]

OptSt= OptiStack(layers, no_back_reflection=False, substrate=Si, incidence=Air)

RAT= calculate_rat(OptSt, options['wavelengths']*1e9, angle=inc_angle, no_back_reflection=False, pol=options['pol'])

Ge_e = np.real((Ge.n(wavelengths) + 1j*Ge.k(wavelengths))**2)
GaAs_e = np.real((GaAs.n(wavelengths) + 1j*GaAs.k(wavelengths))**2)

plt.figure()
plt.plot(wavelengths*1e9, RAT['A_per_layer'][1]/Af_1, label='A1 TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['A_per_layer'][2]/Af_2, label='A2 TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['R']/Rf, label='R TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['T']/Tf, label='T TMM/RCWA')
#plt.plot(wavelengths*1e9, 1/GaAs.n(wavelengths))
#plt.plot(wavelengths*1e9, 1/Ge.n(wavelengths))
#plt.plot(wavelengths*1e9, Af_1, '--', label='Af_1')
#plt.plot(wavelengths*1e9, Af_2, ':', label='Af_2')
plt.legend()
plt.ylim([0,5])
plt.show()



OptSt= OptiStack(layers[::-1], no_back_reflection=False, substrate=Air, incidence=Si)

RAT= calculate_rat(OptSt, options['wavelengths']*1e9, angle=inc_angle, no_back_reflection=False, pol=options['pol'])

Ge_e = np.real((Ge.n(wavelengths) + 1j*Ge.k(wavelengths))**2)
GaAs_e = np.real((GaAs.n(wavelengths) + 1j*GaAs.k(wavelengths))**2)

plt.figure()
plt.plot(wavelengths*1e9, RAT['A_per_layer'][2]/Ab_1, label='A1 TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['A_per_layer'][1]/Ab_2, label='A2 TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['R']/Rb, label='R TMM/RCWA')
plt.plot(wavelengths*1e9, RAT['T']/Tb, label='T TMM/RCWA')
plt.plot(wavelengths*1e9, Si.n(wavelengths))
#plt.plot(wavelengths*1e9, 1/Ge.n(wavelengths))
#plt.plot(wavelengths*1e9, Af_1, '--', label='Af_1')
#plt.plot(wavelengths*1e9, Af_2, ':', label='Af_2')
plt.legend()
plt.ylim([0, 10])
plt.show()


# plt.figure()
# plt.plot(wavelengths*1e9, RAT_flip['R'], 'k-')
# plt.plot(wavelengths*1e9, RAT_flip['T'], 'k--')
# plt.plot(wavelengths*1e9, results_back[2][:, -(angle_index+1)], 'b-')
# plt.plot(wavelengths*1e9, results_back[3][:, -(angle_index+1)], 'b--')
# plt.plot(wavelengths*1e9,  results_back[2][:, -(angle_index+1)]/RAT_flip['R'])
# plt.plot(wavelengths*1e9,  results_back[3][:, -(angle_index+1)]/RAT_flip['T'])
# plt.ylim([0,10])
# plt.plot
# plt.show()

# plt.figure()
#
# plt.plot(wavelengths*1e9, Af_1, '--', label='Af_1')
# plt.plot(wavelengths*1e9, Af_2, ':', label='Af_2')
#
# plt.plot(wavelengths*1e9, Ab_1,  '--', label='Ab_1')
# plt.plot(wavelengths*1e9, Ab_2,  ':',label='Ab_2')
#
# plt.legend()
# plt.show()