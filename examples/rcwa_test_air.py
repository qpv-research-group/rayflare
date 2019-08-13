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
calc = True

import pandas as pd
names = ['GaAs_subs', 'InAlP', 'InGaP']

for name in names:
    pd.read_csv('data/'+name+'.txt', sep='\t')

GaAs = material('GaAs_WVASE')()
Air = material('Air')()

InAlP = material('InAlP_WVASE')()
InGaP = material('InGaP_WVASE')()

Ag = material('Ag_Jiang')()
# Materials for the anti-reflection coating
MgF2 = material('203', nk_db=True)()
#TiO2 = material('TiO2')()
Ta2O5 = material('410', nk_db = True)()
#Si = material('Si')()
SiN = material('321', nk_db = True)()

fign=2
#x = 500
x=1000
# anti-reflection coating



#size = ((x, 0), (0, x))
# The layer with nanoparticles
#struct_mirror = [Layer(si('120nm'), TiO2, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
#  'halfwidths': (210,210), 'angle': 0}])]
req_r = np.sqrt(0.1257*0.5*500**2*np.sin(np.deg2rad(60))/np.pi)

# NP_layer=[Layer(si('50nm'), Ag)]

#solar_cell = SolarCell(grating)

orders = 20
wavelengths = np.linspace(250, 930, 100)*1e-9

options = {'nm_spacing': 0.5,
           'project_name': 'RCWA_test',
           'calc_profile': False,
           'n_theta_bins': 100,
           'c_azimuth': 1e-7,
           'pol': 'u',
           'wavelengths': wavelengths,
           'theta_in': 0, 'phi_in': 0,
           'parallel': True, 'n_jobs': -1,
           'phi_symmetry': np.pi/2,
           }

all_orders = [3, 9, 19, 39, 75, 99, 125, 147]

import seaborn as sns
pal = sns.cubehelix_palette(3, start=.5, rot=-.9)


spect=np.loadtxt('AM0.csv', delimiter=',')
from solcore.interpolate import interp1d
from solcore.constants import q, h, c

f1 = plt.figure(1)
f2 = plt.figure(2)
f3 = plt.figure(3)

AM0 = interp1d(spect[:,0], spect[:,1])
rad = [req_r, 100]
label = ['Hexagonal', 'Square']
EQE_wl = np.linspace(250.1, 929.9, 700)
Jscs = []
for k1, size  in enumerate([((x, 0),(x/2,np.sin(np.pi/3)*x)), ((x,0), (0,x))]):

    if calc:

        grating = [Layer(si('140nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                      'radius':rad[k1], 'angle': 0},
                                                     {'type': 'circle', 'mat': Ag, 'center': (x/2, x/2),
                                                      'radius': rad[k1], 'angle': 0}
                                                     ])]
        solar_cell = SolarCell(grating)
        output = rcwa(solar_cell, size, 30, options, incidence=GaAs, substrate=Ag, only_incidence_angle=True,
                      front_or_rear='front', surf_name='OPTOS')

        #plt.plot(wavelengths, output['A_layer'].todense()[:, 1, :])
        #plt.plot(wavelengths, output['R'][:,0])
        #plt.plot(wavelengths, output['T'][:,0])
        #plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])



    # AM0 spectrum


    from angles import theta_summary, make_angle_vector
    from config import results_path
    from sparse import load_npz
    import xarray as xr
    import os

    normal_inc = output['full_mat'][:,:,0].todense()
    normal_inc_A = output['A_layer'][:,0,0].todense()
    R = output['R'][:,0]
    T = output['T'][:,0]
    _, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                           options['c_azimuth'])

    sum_mat, R, T = theta_summary(normal_inc, angle_vector)


    # plot critical angle in GaAs

    theta_c = np.arcsin(1/GaAs.n(wavelengths))

    Rth = xr.DataArray(normal_inc[:, :options['n_theta_bins']], dims=['Wavelength (nm)', r'$\sin(\theta)$'],
                       coords={'Wavelength (nm)': wavelengths*1e9,
                                r'$\sin(\theta)$': np.sin(angle_vector[:,1])[0:options['n_theta_bins']]})


    # fraction absorbed in single/double pass
    # plt.figure()
    # plt.plot(EQE_wl, 1-np.exp(-GaAs.alpha(EQE_wl*1e-9)*84e-9))
    # plt.plot(EQE_wl, 1-np.exp(-GaAs.alpha(EQE_wl*1e-9)*2*84e-9))
    # plt.xlim([400, 600])
    # plt.show()

    # plt.figure()
    # plt.plot(wavelengths*1e9, normal_inc[:,0])
    # plt.plot(wavelengths*1e9, np.sum(normal_inc[:, 1:], axis=1))
    # plt.show()

    from rigorous_coupled_wave_analysis.rcwa import get_reciprocal_lattice, fold_phi
    rl = get_reciprocal_lattice(size, orders)

    substrate = Ag
    incidence = Air
    layer = GaAs
    theta=0
    phi=0
    phi_sym = options['phi_symmetry']

    wl=wavelengths*1e9

    fi_x = np.real((incidence.n(wl*1e-9) / wl) * np.sin(theta * np.pi / 180) *
                   np.sin(phi* np.pi / 180))
    fi_y = np.real((incidence.n(wl*1e-9) / wl) * np.sin(theta * np.pi / 180) *
                   np.cos(phi * np.pi / 180))

    xy = np.arange(-2, 3, 1)

    xv, yv = np.meshgrid(xy, xy)
    # print('inc', fi_x, fi_y)

    fr_x = np.add(fi_x[:,None], xv.flatten() * rl[0][0] + yv.flatten() * rl[1][0])
    fr_y = np.add(fi_y[:,None], xv.flatten() * rl[0][1]+ yv.flatten()* rl[1][1])

    eps_inc = (layer.n(wl*1e-9) + 1j*layer.k(wl*1e-9))**2

    eps_sub = (substrate.n(wl*1e-9) + 1j*substrate.k(wl*1e-9))**2
    # print('eps/lambda', l_oc[0]/(wl**2))
    fr_z = np.sqrt((eps_inc/ (wl ** 2))[:,None] - fr_x ** 2 - fr_y ** 2)

    ft_z = np.sqrt((eps_sub / (wl ** 2))[:,None] - fr_x ** 2 - fr_y ** 2)

    # print('ref', fr_x, fr_y, fr_z)

    phi_rt = np.nan_to_num(np.arctan(fr_x / fr_y))
    phi_rt = fold_phi(phi_rt, phi_sym)
    theta_r = np.real(np.arccos(fr_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + fr_z ** 2)))
    theta_t = np.pi - np.real(np.arccos(ft_z / np.sqrt(fr_x ** 2 + fr_y ** 2 + ft_z ** 2)))

    plt.figure(fign)
    ax = plt.subplot(111)
    ax = Rth.plot.imshow(ax=ax)
    plt.plot(theta_c, wavelengths*1e9)
    # ax.set_clim([0, 0.5])
    plt.plot(np.sin(np.unique(theta_r,axis=1)), wl, '--', color='white', alpha=0.2)

    fign = fign+1

plt.figure(1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorpion in GaAs')
plt.ylim([0,1])
plt.show()

plt.figure(2)
plt.show()

plt.figure(3)
plt.show()