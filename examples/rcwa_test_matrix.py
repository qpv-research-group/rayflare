import numpy as np
import matplotlib.pyplot as plt

from solcore import si, material
from solcore.structure import Junction, Layer
from solcore.solar_cell import SolarCell
from solcore.solar_cell_solver import solar_cell_solver, default_options
from solcore.light_source import LightSource
from rigorous_coupled_wave_analysis.rcwa import rcwa
from solcore.absorption_calculator.nk_db import download_db, search_db
from solcore.material_system.create_new_material import create_new_material
import xarray as xr

from angles import make_angle_vector, theta_summary

GaAs = material('GaAs')()
Air = material('Air')()

Ag = material('Ag_Jiang')()
# Materials for the anti-reflection coating
MgF2 = material('203', nk_db=True)()
#TiO2 = material('TiO2')()
Ta2O5 = material('410', nk_db = True)()
#Si = material('Si')()
SiN = material('SiN', nk_db = True)()

#x = 500
x=500
# anti-reflection coating

ARC = [Layer(si('78nm'), MgF2), Layer(si('48nm'), Ta2O5)]

size = ((x, 0),(x/2,np.tan(np.pi/3)*x))
# The layer with nanoparticles
#struct_mirror = [Layer(si('120nm'), TiO2, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
                                                   #  'halfwidths': (210,210), 'angle': 0}])]

grating =  [Layer(si('120nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                     'radius': 100, 'angle': 0}])]
# NP_layer=[Layer(si('50nm'), Ag)]


solar_cell = SolarCell(ARC + [Layer(material=GaAs, width=si('80nm'))] + grating)

#solar_cell = SolarCell(grating)

orders = 20
wavelengths = np.linspace(250, 930, 40)*1e-9

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

all_orders = [5, 10, 20, 40, 75, 100]#, 100, 150]
import seaborn as sns
pal = sns.cubehelix_palette(len(all_orders), start=.5, rot=-.9)


spect=np.loadtxt('AM0.csv', delimiter=',')
from solcore.interpolate import interp1d
from solcore.constants import q, h, c

AM0 = interp1d(spect[:,0], spect[:,1])

EQE_wl = np.linspace(250.1, 929.9, 700)
Jscs = []

plt.figure()
for i1, orders in enumerate(all_orders):
    print(orders)
    grating = [Layer(si('120nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                  'radius': 100, 'angle': 45}])]
    solar_cell = SolarCell(ARC + [Layer(material=GaAs, width=si('80nm'))] + grating)
    output = rcwa(solar_cell, size, orders, options, incidence=Air, substrate=Ag, only_incidence_angle=True,
                           front_or_rear='front', surf_name='OPTOS')

    A_GaAs = interp1d(wavelengths * 1e9, output['A_layer'].todense()[:, 2, :][:, 0])
    Jsc = 0.1 * (q / (h * c)) * np.trapz(EQE_wl * A_GaAs(EQE_wl) * AM0(EQE_wl), EQE_wl) / 1e9
    Jscs.append(Jsc)
    #plt.figure()
    #plt.plot(wavelengths, output['A_layer'].todense()[:, 0, :])
    #plt.plot(wavelengths, output['A_layer'].todense()[:, 1, :])
    plt.plot(wavelengths, output['A_layer'].todense()[:, 2, :], color=pal[i1], label=str(orders))
    plt.plot(wavelengths, output['A_layer'].todense()[:, 3, :], '--', color=pal[i1], label=str(orders))
    to_save = np.vstack((wavelengths, output['A_layer'].todense()[:, 2, :][:, 0])).T
    np.savetxt('A_GaAs_' + str(orders) +'.csv', to_save, delimiter=',')
    #plt.plot(wavelengths, output['R'][:,0])
    #plt.plot(wavelengths, output['T'][:,0])
    #plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])

plt.legend()
plt.show()


# AM0 spectrum
plt.figure()
plt.plot(all_orders, Jscs)
plt.show()


plt.figure()
for i1, orders in enumerate(all_orders):
    A_GaAs = np.loadtxt('A_GaAs_' + str(orders) +'.csv', delimiter=',')
    plt.plot(A_GaAs[:,0]*1e9, A_GaAs[:,1], color=pal[i1], label=str(orders))
    #plt.plot(wavelengths, output['T'][:,0])
    #plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])

plt.legend()
plt.show()

Jscs = []
orders = 40
radii = [50, 100, 150, 200, 250]

plt.figure()
for i1, rad in enumerate(radii):
    print(orders)
    grating = [Layer(si('120nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                  'radius': rad, 'angle': 45}])]
    solar_cell = SolarCell(ARC + [Layer(material=GaAs, width=si('80nm'))] + grating)
    output = rcwa(solar_cell, size, orders, options, incidence=Air, substrate=Ag, only_incidence_angle=True,
                           front_or_rear='front', surf_name='OPTOS')

    A_GaAs = interp1d(wavelengths * 1e9, output['A_layer'].todense()[:, 2, :][:, 0])
    Jsc = 0.1 * (q / (h * c)) * np.trapz(EQE_wl * A_GaAs(EQE_wl) * AM0(EQE_wl), EQE_wl) / 1e9
    Jscs.append(Jsc)
    #plt.figure()
    #plt.plot(wavelengths, output['A_layer'].todense()[:, 0, :])
    #plt.plot(wavelengths, output['A_layer'].todense()[:, 1, :])
    plt.plot(wavelengths, output['A_layer'].todense()[:, 2, :], color=pal[i1], label=str(rad))
    plt.plot(wavelengths, output['A_layer'].todense()[:, 3, :], '--', color=pal[i1], label=str(rad))
    to_save = np.vstack((wavelengths, output['A_layer'].todense()[:, 2, :][:, 0])).T
    np.savetxt('A_GaAs_rad_' + str(rad) +'.csv', to_save, delimiter=',')
    #plt.plot(wavelengths, output['R'][:,0])
    #plt.plot(wavelengths, output['T'][:,0])
    #plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])

plt.legend()
plt.show()

plt.figure()
plt.plot(radii, Jscs)
plt.show()
