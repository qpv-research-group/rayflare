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
calc = False


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
x=500
# anti-reflection coating

ARC = [Layer(si('60nm'), SiN)]

#size = ((x, 0), (0, x))
# The layer with nanoparticles
#struct_mirror = [Layer(si('120nm'), TiO2, geometry=[{'type': 'rectangle', 'mat': Ag, 'center': (x/2, x/2),
#  'halfwidths': (210,210), 'angle': 0}])]
req_r = np.sqrt(0.1257*0.5*500**2*np.sin(np.deg2rad(60))/np.pi)

# NP_layer=[Layer(si('50nm'), Ag)]

#solar_cell = SolarCell(grating)

orders = 20
wavelengths = np.linspace(250, 930, 80)*1e-9

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
ropt = dict(LatticeTruncation = 'Circular',
            DiscretizedEpsilon = True,
            DiscretizationResolution = 4,
            PolarizationDecomposition = False,
            PolarizationBasis = 'Default',
            LanczosSmoothing = True,
            SubpixelSmoothing = True,
            ConserveMemory = False,
            WeismannFormulation = True)
options['rcwa_options'] = ropt

import seaborn as sns



spect=np.loadtxt('AM0.csv', delimiter=',')
from solcore.interpolate import interp1d
from solcore.constants import q, h, c

plt.figure()
orders= np.arange(10, 60, 10)
np.insert(orders, values=0, obj=0)
pal = sns.cubehelix_palette(len(orders), start=.5, rot=-.9)


AM0 = interp1d(spect[:,0], spect[:,1])
rad = [req_r, 100]
label = ['Hexagonal', 'Square']
EQE_wl = np.linspace(250.1, 929.9, 700)
Jscs = []
for k1, size  in enumerate([((x, 0),(x/2,np.sin(np.pi/3)*x))]):
    for l1, no in enumerate(orders):
        if calc:
            grating = [Layer(si(120, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                          'radius':rad[k1], 'angle': 0}])]
            solar_cell = SolarCell(ARC + [Layer(material=InGaP, width=si('19nm')),
                Layer(material=GaAs, width=si('86nm')),
                                          Layer(material=InAlP, width=si('18nm'))] +
                                   grating, substrate=Ag)
            output = rcwa(solar_cell, size, no, options, incidence=Air, substrate=Ag, only_incidence_angle=True,
                          front_or_rear='front', surf_name='OPTOS', detail_layer=3)

            A_GaAs = interp1d(wavelengths * 1e9, output['A_layer'].todense()[:, 2, :][:, 0], kind=2)
            A_InGaP = interp1d(wavelengths * 1e9, output['A_layer'].todense()[:, 1, :][:, 0], kind=2)
            A_InAlP = interp1d(wavelengths * 1e9, output['A_layer'].todense()[:, 3, :][:, 0], kind=2)
            R = interp1d(wavelengths *1e9, output['R'][:,0])
            Jsc = 0.1 * (q / (h * c)) * np.trapz(EQE_wl * A_GaAs(EQE_wl) * AM0(EQE_wl), EQE_wl) / 1e9
            Jscs.append(Jsc)

            plt.plot(EQE_wl, A_GaAs(EQE_wl), label=str(no), color=pal[l1])
            plt.plot(EQE_wl, R(EQE_wl), color=pal[l1])
            plt.plot(EQE_wl, A_InGaP(EQE_wl), '--', color=pal[l1])
            plt.plot(EQE_wl, A_InAlP(EQE_wl), '-.', color=pal[l1])

            save_name = 'realstruct_' + str(no) + '.csv'
            to_save = np.vstack((wavelengths, output['A_layer'].todense()[:, 1, :][:, 0],
                                 output['A_layer'].todense()[:, 2, :][:, 0],
                                 output['A_layer'].todense()[:, 3, :][:, 0],
                                 output['A_layer'].todense()[:, 4, :][:, 0],
                                 output['R'][:, 0],
                                 output['T'][:, 0])).T
            np.savetxt('convergence_tests/' + save_name, to_save, delimiter=',')
            #plt.plot(wavelengths, output['A_layer'].todense()[:, 1, :])
            #plt.plot(wavelengths, output['R'][:,0])
            #plt.plot(wavelengths, output['T'][:,0])
            #plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])

        else:
            save_name = 'realstruct_' + str(no) + '.csv'

            allA = np.loadtxt('convergence_tests/' + save_name, delimiter=',')

            A_GaAs = interp1d(wavelengths * 1e9, allA[:,2], kind=2)
            A_InGaP = interp1d(wavelengths * 1e9, allA[:,1], kind=2)
            A_InAlP = interp1d(wavelengths * 1e9, allA[:,3], kind=2)
            A_mirror = interp1d(wavelengths * 1e9, allA[:,4], kind=2)
            R = interp1d(wavelengths * 1e9, allA[:,4])
            Jsc = 0.1 * (q / (h * c)) * np.trapz(EQE_wl * A_GaAs(EQE_wl) * AM0(EQE_wl), EQE_wl) / 1e9
            Jscs.append(Jsc)

            plt.plot(EQE_wl, A_GaAs(EQE_wl), label=str(no), color=pal[l1])
            plt.plot(EQE_wl, A_InAlP(EQE_wl)+A_GaAs(EQE_wl)+A_InGaP(EQE_wl)+A_mirror(EQE_wl), '--', color=pal[l1])
            #plt.plot(EQE_wl, R(EQE_wl), color=pal[l1])
            #plt.plot(EQE_wl, A_InGaP(EQE_wl), '--', color=pal[l1])
            #plt.plot(EQE_wl, A_InAlP(EQE_wl), '-.', color=pal[l1])

            # plt.plot(wavelengths, output['A_layer'].todense()[:, 1, :])
            # plt.plot(wavelengths, output['R'][:,0])
            # plt.plot(wavelengths, output['T'][:,0])
            # plt.legend(['ARC1', 'ARC2', 'GaAS', 'grting', 'R', 'T'])


from solcore.solar_cell_solver import solar_cell_solver
solar_cell_solver(solar_cell, 'optics', {'wavelength': EQE_wl*1e-9, 'optics_method': 'TMM',
                                         'no_back_reflexion': False})

plt.plot(EQE_wl, solar_cell[2].layer_absorption, label='TMM')
plt.show()

plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorpion in GaAs')
plt.ylim([0,1])
plt.show()

plt.figure()
plt.plot(orders, Jscs)
plt.show()