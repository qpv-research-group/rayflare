import numpy as np
import matplotlib.pyplot as plt

from solcore import si, material
from solcore.structure import Layer
from solcore.solar_cell import SolarCell
from rigorous_coupled_wave_analysis.rcwa import rcwa
from solcore.interpolate import interp1d
from solcore.constants import q, h, c
from solcore.solar_cell_solver import solar_cell_solver
import xarray as xr
from sparse import save_npz, load_npz
from cycler import cycler
from analytic.diffraction import get_order_directions
import seaborn as sns

from angles import theta_summary, make_angle_vector
import xarray as xr

calc = True
all_orders = [55]
# all_orders = [109, 121, 127, 139, 151]

wavelengths = np.linspace(250, 1100, 200) * 1e-9
EQE_wl = np.linspace(250.1, 1099.9, 700)
wl = wavelengths * 1e9

RCWA_wl = wavelengths[:]
## find wavelength below which absorption in 86 nm > 99 %


from analytic.diffraction import get_order_directions

rads = np.arange(100, 131, 5)

#rads = [rads[2]]


mirror = [106.0]

pal = sns.cubehelix_palette(len(mirror), start=.5, rot=-.9)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

all_R0 = xr.DataArray(np.zeros((len(all_orders), len(mirror), len(rads), len(wl))),
                       dims = ['orders', 'mirror_th', 'rads', 'wl'],
                      coords={'wl': wl, 'mirror_th': mirror, 'rads': rads})

all_Rdiff = xr.DataArray(np.zeros((len(all_orders), len(mirror), len(rads), len(wl))),
                       dims = ['orders', 'mirror_th', 'rads', 'wl'],
                      coords={'wl': wl, 'mirror_th': mirror, 'rads': rads})

all_R_TMM = xr.DataArray(np.zeros((len(mirror), len(wl))),
                       dims = ['mirror_th', 'wl'],
                      coords={'wl': wl, 'mirror_th': mirror})

pal = sns.cubehelix_palette(len(mirror), start=.5, rot=-.9)

for i1, orders in enumerate(all_orders):
    GaAs = material('GaAs_WVASE')()
    Air = material('Air')()

    InAlP = material('InAlP_WVASE')()
    InGaP = material('InGaP_WVASE')()
    Ag = material('Ag_Jiang')()

    #SiN = material('321', nk_db = True)()
    SiN = material('SiN_SE')()
    theta_c = np.arcsin(1 / GaAs.n(wavelengths))

    x=500
    # anti-reflection coating
    size = ((x, 0),(x/2,np.sin(np.pi/3)*x))



    options = {'nm_spacing': 0.5,
               'n_theta_bins': 100,
               'c_azimuth': 0.1,
               'pol': 'u',
               'wavelengths': RCWA_wl,
               'theta_in': 0, 'phi_in': 0,
               'parallel': True, 'n_jobs': -1,
               'phi_symmetry': np.pi/2,
               'project_name': 'ultrathin'
               }

    anlt = get_order_directions(wl, size, 3, Air, InAlP, 0, 0, options['phi_symmetry'])

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


    weighted_Rtot = np.zeros((len(mirror), len(rads)))

    weighted_diff = np.zeros((len(mirror), len(rads)))
    weighted_Amirr = np.zeros((len(mirror), len(rads)))


    pal = sns.cubehelix_palette(len(rads), start=.5, rot=-.9)

    linestyle = ['solid', (0, (1, 10)), (0, (1, 1)),(0, (1, 1)),
    (0, (5, 10)),(0, (5, 5)),(0, (5, 1)),(0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)),
         (0, (3, 1, 1, 1)),(0, (3, 5, 1, 5, 1, 5)), (0, (3, 10, 1, 10, 1, 10)),
                       (0, (3, 1, 1, 1, 1, 1))]

    plt.figure()

    for l1, mirror_thick in enumerate(mirror):
        grating2 = [Layer(si(mirror_thick, 'nm'), SiN)]
        solar_cell = SolarCell([Layer(material=GaAs, width=si('25nm')),
                                      Layer(material=InGaP, width=si('19nm')),
                                          Layer(material=GaAs, width=si('86nm')),
                                          Layer(material=InAlP, width=si('18nm'))] + grating2,
                               substrate=Ag)
        solar_cell_solver(solar_cell, 'optics', {'wavelength': wavelengths, 'optics_method': 'TMM',
                                                 'no_back_reflexion': False,
                                                 'recalculate_absorption': True})
        all_R_TMM[l1] = solar_cell.reflected
        for k1, rad in enumerate(rads):
            if calc:
                grating1 = [Layer(si(mirror_thick/2, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Air, 'center': (0, 0),
                                                              'radius': rad, 'angle': 0}])]
                grating2 = [Layer(si(mirror_thick/2, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                              'radius': rad, 'angle': 0}])]
                solar_cell = SolarCell([Layer(material=GaAs, width=si('25nm')),
                                           Layer(material=InGaP, width=si('19nm')),
                                           Layer(material=GaAs, width=si('86nm')),
                                           Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                                       substrate=Ag)
                output = rcwa(solar_cell, size, orders, options, incidence=Air, substrate=GaAs,
                              only_incidence_angle=True,
                              front_or_rear='front', surf_name='OPTOS', detail_layer=False)


                R_comb = np.hstack(output['R'][:, 0])
                #to_save = np.vstack((wavelengths, A_GaAs_comb, A_InGaP_comb, A_InAlP_comb,
                #                     A_mirror_comb, R_comb)).T
                #np.savetxt('results/mirror_' + str(mirror_thick) +'_diskr_'+str(rad) +  '_orders_' + str(orders) + '.csv', to_save, delimiter=',')

                #A_GaAs = interp1d(wavelengths*1e9, A_GaAs_comb)
                intmat = output['full_mat']
                #save_npz('results/GaAsinc_mirror_' + str(mirror_thick) +'_diskr_'+str(rad) +  '_orders_' + str(orders) + '.npz', intmat)
                #Jsc = 0.1 * (q / (h * c)) * np.trapz(EQE_wl * A_GaAs(EQE_wl) * AM0(EQE_wl), EQE_wl) / 1e9


            else:

                res = np.loadtxt('results/mirror_' + str(mirror_thick) +'_diskr_'+str(rad) +  '_orders_' + str(orders) + '.csv', delimiter=',')

                R_comb = res[:, 5]
                A_mirror_comb = res[:, 4]
                A_GaAs = interp1d(wavelengths*1e9, res[:,1])

                intmat = load_npz('results/GaAsinc_mirror_' + str(mirror_thick) +'_diskr_'+str(rad) +  '_orders_' + str(orders) + '.npz')


            normal_inc = intmat[:,:,0].todense()
            normal_inc = normal_inc[:,:(options['n_theta_bins']+1)]
            theta_intv, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                                   options['c_azimuth'])

            # plot critical angle in GaAs

            R0 = normal_inc[:,0]
            Rdiff = np.sum(normal_inc[:,1:options['n_theta_bins']], axis=1)
            Rtot = R0 + Rdiff

            Rth = xr.DataArray(normal_inc[:, :options['n_theta_bins']], dims=['Wavelength (nm)', r'$\sin(\theta)$'],
                               coords={'Wavelength (nm)': RCWA_wl * 1e9,
                                       r'$\sin(\theta)$': np.sin(angle_vector[:, 1])[0:options['n_theta_bins']]})

            all_R0[i1, l1, k1] = R0
            all_Rdiff[i1, l1, k1] = Rdiff

max_order = len(all_orders) - 1

mean_R = all_R0.reduce(np.mean, dim='orders')
sd_R = all_R0.reduce(np.std, dim='orders')


grating = [Layer(si(mirror_thick, 'nm'), Ag)]
solar_cell = SolarCell([Layer(material=GaAs, width=si('25nm')),
                        Layer(material=InGaP, width=si('19nm')),
                        Layer(material=GaAs, width=si('86nm')),
                        Layer(material=InAlP, width=si('18nm'))] + grating,
                       substrate=Ag)
solar_cell_solver(solar_cell, 'optics', {'wavelength': wavelengths, 'optics_method': 'TMM',
                                         'no_back_reflexion': False,
                                         'recalculate_absorption': True})
all_R_TMM_Ag = solar_cell.reflected
pal = sns.cubehelix_palette(4, start=.5, rot=-.9)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

plt.figure()
all_R_TMM.plot.line(x='wl', label='TMM - SiN layer')
plt.plot(wl, all_R_TMM_Ag, label='TMM - Ag layer')
(all_R0[0, 0, 3] + all_Rdiff[0,0,3]).plot.line(x='wl', label='RCWA')
plt.legend()
plt.xlim(300, 1100)
plt.show()
