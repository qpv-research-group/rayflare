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


wavelengths = np.linspace(300, 1200, 50)*1e-9

options = default_options
options['n_theta_bins'] = 100
options['wavelengths'] = wavelengths
options['c_azimuth'] = 0.0001
options['phi_symmetry'] = np.pi/2


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


problem = TMM(layers, Air, Si, 'test', options, True, None, [], 'front', False)

full = problem[0].todense()[4]



summat = theta_summary(full, angle_vector, options['n_theta_bins'])

summat = xr.DataArray(summat, dims=[r'$\theta_{out}$', r'$\theta_{in}$'], coords={r'$\theta_{out}$': theta_all, r'$\theta_{in}$': theta_all})
#
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

full_back = problem_back[0].todense()[4]



summat_back = theta_summary(full_back, angle_vector, options["n_theta_bins"])


#summat_back = xr.DataArray(summat_back, dims=[r'$\theta_{out}$', r'$\theta_{in}$'], coords={r'$\theta_{out}$': theta_all, r'$\theta_{in}$': theta_all})
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

whole_mat_imshow = whole_mat_imshow.interp(theta_in = np.linspace(0, np.pi, 100), theta_out =  np.linspace(0, np.pi, 100))

whole_mat_imshow = whole_mat_imshow.rename({'theta_in': r'$\theta_{in}$', 'theta_out' : r'$\theta_{out}$'})

fig = plt.figure()
ax = plt.subplot(111)
ax = whole_mat_imshow.plot.imshow(ax=ax, cmap=seamap)
#ax = plt.subplot(212)
fig.savefig('matrix.png', bbox_inches='tight', format='png')
#ax = Tth.plot.imshow(ax=ax)

plt.show()

from sparse import nansum

# indexing: wl, th_out, th_in
Rf = nansum(problem[0][:, 0:(int(len(angle_vector)/2))], 1)[:, 10].todense()
Tf = nansum(problem[0][:, (int(len(angle_vector)/2)):], 1)[:, 10].todense()
Af_1 = problem[2].Alayer[0, :, 10, 0]
Af_2 = problem[2].Alayer[0, :, 10, 1]

Tb = nansum(problem_back[0][:, 0:(int(len(angle_vector)/2))], 1)[:, len(angle_vector)-11].todense()
Rb = nansum(problem_back[0][:, (int(len(angle_vector)/2)):], 1)[:, len(angle_vector)-11].todense()
Ab_1 = problem_back[2].Alayer[0,:,-11,0]
Ab_2 = problem_back[2].Alayer[0,:,-11,1]


plt.figure()
plt.plot(wavelengths*1e9, Rf, label='Rf')
plt.plot(wavelengths*1e9, Tf, '-.', label='Tf')
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
