import numpy as np
import matplotlib.pyplot as plt

from solcore import si, material
from solcore.structure import Junction, Layer
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver, default_options
from solcore.light_source import LightSource
from rigorous_coupled_wave_analysis.rcwa import calculate_rat_rcwa
from solcore.absorption_calculator.nk_db import download_db, search_db
from solcore.material_system.create_new_material import create_new_material
import xarray as xr

from angles import make_angle_vector, theta_summary

GaAs = material('GaAs')()
Air = material('Air')()
TiO2 = material('TiO2', sopra=True)()  # for the nanoparticles
Ag = material('Ag_J')()
# Materials for the anti-reflection coating
MgF2 = material('MgF2')()
TiO2 = material('TiO2')()
Ta2O5 = material('410', nk_db = True)()
Ag = material('Ag_J')()
Si = material('Si')()

#x = 500
x=1000
# anti-reflection coating
ARC = [Layer(si('78nm'), material = MgF2), Layer(si('48nm'), material = Ta2O5)]

size = ((x, 0),(0,x))
# The layer with nanoparticles
#struct_mirror = [Layer(si('120nm'), TiO2, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
                                                   #  'halfwidths': (210,210), 'angle': 0}])]

grating =  [Layer(si('120nm'), Si, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
                                                     'halfwidths': (np.sqrt(2*(500**2))/2, np.sqrt(2*(500**2))/2), 'angle': 45}])]
# NP_layer=[Layer(si('50nm'), Ag)]


#solar_cell = SolarCell(ARC + [Layer(material=GaAs, width=si('300nm'))] + struct_mirror)

solar_cell = SolarCell(grating)

orders = 70
wavelengths = np.linspace(600, 1100, 4)*1e-9

options = {'nm_spacing': 0.5,
           'project_name': 'RCWA_test',
           'calc_profile': False,
           'n_theta_bins': 15,
           'c_azimuth': 0.25,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 0, 'phi_in': 0,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           }

output = calculate_rat_rcwa(solar_cell, size, orders, options, incidence=Si, substrate=Air, only_incidence_angle=False,
                       front_or_rear='front', surf_name='OPTOS')

#plt.plot(wavelengths, output['A_layer'][:, 0, 0])
#plt.plot(wavelengths, output['A_layer'][:, 0, 1])
#plt.plot(wavelengths, output['A_layer'][:, 0, 2])
#plt.plot(wavelengths, output['A_layer'][:, 0, 3])
plt.plot(wavelengths, output['R'][:,0])
plt.plot(wavelengths, output['T'][:,0])
plt.legend(['ARC1', 'ARC2', 'GaAs', 'struct mirror', 'R', 'T'])
plt.show()
#
# #plt.figure()
# #plt.plot(output['theta_r'][0], output['R_pfbo'][0], 'o')
# #plt.plot(output['theta_r'][0], output['R_pfbo_2'][0], 'o')
#
#
# theta_intv, phi_intv, angle_vector = make_angle_vector(20, np.pi/2, 0.25)
#
# binned_theta_out, _ = np.histogram(output['theta_r'][0], bins=theta_intv, weights=output['R_pfbo'][0])
# binned_theta_t, _ = np.histogram(output['theta_t'][0], bins=theta_intv, weights=output['T_pfbo'][0])
# plt.figure()
# plt.plot(binned_theta_out, 'o')
# plt.plot(binned_theta_t, 'o')
#
# print(output['T'][7])
# print(np.sum(output['T_pfbo'][7]))

theta_intv, phi_intv, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'], options['c_azimuth'])

R_mat = output['R_mat'][3]
T_mat = output['T_mat'][3]

fullmat = np.hstack((R_mat, T_mat))

sum_mat, R, T = theta_summary(R_mat, angle_vector)

sum_mat = xr.DataArray(sum_mat, dims=['sin_out', 'sin_in'], coords={'sin_out': np.linspace(0,1,options['n_theta_bins']),
                       'sin_in': np.linspace(0,1,options['n_theta_bins'])})

fig = plt.figure()
ax = plt.subplot(111)
sum_mat.plot.imshow(ax=ax, vmax=5)



