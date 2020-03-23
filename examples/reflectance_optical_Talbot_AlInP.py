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
all_orders = [5]
# all_orders = [109, 121, 127, 139, 151]

wavelengths = np.linspace(250, 1100, 200) * 1e-9
EQE_wl = np.linspace(250.1, 1099.9, 700)
wl = wavelengths * 1e9

RCWA_wl = wavelengths[:]
## find wavelength below which absorption in 86 nm > 99 %


from analytic.diffraction import get_order_directions

rads = np.arange(100, 131, 5)

#rads = [rads[2]]

mirror = np.arange(60, 101, 5)

#mirror = [mirror[2]]

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
        grating = [Layer(si(mirror_thick, 'nm'), SiN)]
        solar_cell = SolarCell(grating + [Layer(material=InAlP, width=si('18nm')),
                                          Layer(material=GaAs, width=si('86nm')),
                                          Layer(material=InGaP, width=si('19nm'))],
                               substrate=GaAs)
        solar_cell_solver(solar_cell, 'optics', {'wavelength': wavelengths, 'optics_method': 'TMM',
                                                 'no_back_reflexion': False,
                                                 'recalculate_absorption': True})
        all_R_TMM[l1] = solar_cell.reflected
        for k1, rad in enumerate(rads):
            if calc:
                grating = [Layer(si(mirror_thick, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Air, 'center': (0, 0),
                                                              'radius': rad, 'angle': 0}])]
                solar_cell = SolarCell(grating + [Layer(material=InAlP, width=si('18nm')),
                                                  Layer(material=GaAs, width=si('86nm')),
                                                  Layer(material=InGaP, width=si('19nm'))],
                                       substrate=GaAs)
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

plt.figure()
all_R0[:, 0, 0].plot.line(x='wl')
plt.show()

plt.figure()
all_Rdiff[:, 0, 0].plot.line(x='wl')
plt.show()

plt.figure()
#all_R0[4, 2, 2].plot.line()
#all_Rdiff[4, 2, 2].plot.line()
(all_Rdiff[max_order, 2, 2]/all_R0[max_order, 2, 2]).plot.line()
plt.show()

plt.figure()
all_R0[max_order, 2, :].plot.imshow()

plt.show()


plt.figure()
all_R0[max_order, :, 2].plot.imshow()

plt.show()

for i1 in np.arange(0,all_R0.shape[1]):
    plt.figure()
    (all_R0[max_order, i1, :] + all_Rdiff[max_order,i1,:]).plot.line(x='wl')
    plt.xlim(300, 900)
    plt.show()


# SPIN
plt.figure()
(all_R0[max_order, :, 0] + all_Rdiff[max_order, :,2]).plot.line(x='wl')
plt.xlim(300,1100)
plt.ylabel('SPIN')
plt.ylim(0, 0.4)
plt.show()


#SPEX
plt.figure()
all_Rdiff[max_order, :, 0].plot.line(x='wl')
plt.xlim(300, 600)
plt.ylabel('SPEX')
plt.ylim(0, 0.2)
plt.show()
#
# mean_A = all_A.reduce(np.mean, dim='orders')
#
# mean_R = mean_R.assign_coords(rads=rads, mirror_th=mirror)
# mean_A = mean_A.assign_coords(rads=rads, mirror_th=mirror)
#
# mean_diff = all_diff.reduce(np.mean, dim='orders')
# mean_diff = mean_diff.assign_coords(rads=rads, mirror_th=mirror)
# sd_diff = all_diff.reduce(np.std, dim='orders')
#
# plt.figure()
# plt.plot(rads, mean_R.T)
# plt.legend([str(x) for x in mirror], title='Mirror thickness (nm)')
# plt.xlabel('Disk radius (nm)')
# plt.show()
#
# plt.figure()
# plt.plot(rads, sd_R.T)
# plt.legend([str(x) for x in mirror])
# plt.show()
#
# plt.figure()
# plt.plot(rads, mean_diff.T)
# plt.legend([str(x) for x in mirror], title='Mirror thickness (nm)')
# plt.xlabel('Disk radius (nm)')
# plt.show()
#
# plt.figure()
# plt.plot(rads, sd_diff.T)
# plt.legend([str(x) for x in mirror])
# plt.show()
#
#
# plt.figure()
# mean_R.plot.imshow()
# plt.xlabel('Disk radius (nm)')
# plt.ylabel('Mirror thickness (nm)')
# plt.title('Total reflection')
# plt.show()
#
plt.figure()
all_R_TMM.plot.line(x='wl')
plt.show()

grating = [Layer(si(mirror_thick, 'nm'), SiN)]
solar_cell = SolarCell([Layer(material=InAlP, width=si('18nm')),
                                  Layer(material=GaAs, width=si('86nm')),
                                  Layer(material=InGaP, width=si('19nm'))],
                       substrate=GaAs)
solar_cell_solver(solar_cell, 'optics', {'wavelength': wavelengths, 'optics_method': 'TMM',
                                         'no_back_reflexion': False,
                                         'recalculate_absorption': True})

anlt = get_order_directions(wl, size, 3, Air, GaAs, 0, 0, options['phi_symmetry'])



plot_angles_500 = anlt['theta_r'][:, 16]



f = plt.figure()
plt.plot(wl, plot_angles_500*180/np.pi, '-')
plt.plot(0, 0, '-k')
plt.plot(0, 0, '--k')

#plt.plot(wl, theta_c, '-r')
plt.xlim(250, 1050)
plt.ylim(0, 90)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Angle ($^\circ$)')
plt.legend(['(1, 0)', '(1, 2)', '(0, 2)', '(-1, 2)', '(-2, 2)', 'Period = 500 nm', 'Period = 900 nm'])
plt.show()