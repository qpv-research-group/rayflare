# try optimizing an anti-reflection coating

import numpy as np
import seaborn as sns
from cycler import cycler

from solcore import si, material
from solcore.structure import Layer
from rigorous_coupled_wave_analysis.rcwa import rcwa_structure

import matplotlib.pyplot as plt

# Import the DE implementations
from solcore.optimization import PDE, DE


from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell
from solcore.constants import q, h, c

pal = sns.color_palette("hls", 3)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols
plt.rcParams['font.size'] = 11

all_orders = [7, 19, 37, 43, 55, 73, 85, 97, 109]

use_orders = [37, 43, 55, 73, 85, 97, 109]

params = np.zeros((len(use_orders), 5))

calc = False

# define the problem
class EQE_fit:
    def __init__(self, orders):


        x = 500
        self.orders = orders

        EQE_data = np.loadtxt('data/V2A_9_EQE.dat', skiprows=129)


        # anti-reflection coating
        #ARC1 = [Layer(si('60nm'), SiN)]
        self.size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))


        wavelengths = np.linspace(303, 1000, 160) * 1e-9


        RCWA_wl = wavelengths

        options = {'nm_spacing': 0.5,
                   'n_theta_bins': 100,
                   'c_azimuth': 1e-7,
                   'pol': 's',
                   'wavelengths': RCWA_wl,
                   'theta_in': 0, 'phi_in': 0,
                   'parallel': True, 'n_jobs': -1,
                   'phi_symmetry': np.pi / 2,
                   'project_name': 'ultrathin'
                   }

        ropt = dict(LatticeTruncation='Circular',
                    DiscretizedEpsilon=True,
                    DiscretizationResolution=4,
                    PolarizationDecomposition=False,
                    PolarizationBasis='Default',
                    LanczosSmoothing=True,
                    SubpixelSmoothing=True,
                    ConserveMemory=False,
                    WeismannFormulation=True)
        options['rcwa_options'] = ropt
        self.options = options

        spect = np.loadtxt('AM0.csv', delimiter=',')
        self.AM0 = interp1d(spect[:, 0], spect[:, 1])(wavelengths*1e9)
        self.EQE_meas = interp1d(EQE_data[:,0], EQE_data[:,1])(wavelengths*1e9)

    def evaluate(self, x):
        GaAs = material('GaAs_WVASE')()
        Air = material('Air')()

        InAlP = material('InAlP_WVASE')()
        InGaP = material('InGaP_WVASE')()

        Ag = material('Ag_Jiang')()

        # MgF2 = material('MgF2')()
        # Ta2O5 = material('410', nk_db=True)()
        SiN = material('SiN_SE')()
        # print('SiN thickness', x[2]*x[1], 'Ag th', (1-x[1])*x[2])

        grating1 = [Layer(si(x[2]*x[1], 'nm'), SiN)]

        grating2 = [Layer(si((1-x[1])*x[2], 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                        'radius': x[3], 'angle': 0}])]

        solar_cell = SolarCell([Layer(material=InGaP, width=si('19nm')),
                                       Layer(material=GaAs, width=si(x[0], 'nm')),
                                       Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                               substrate=Ag)

        S4_setup = rcwa_structure(solar_cell, self.size, self.orders, self.options, Air, Ag)

        RAT = S4_setup.calculate()
        EQE_sim = 0.9*100*(RAT['A_layer'][:,0] + RAT['A_layer'][:,1])

        # least squares?
        residual = np.sum((EQE_sim-self.EQE_meas)**2)


        return residual

    def plot(self, x, icol=0):

        if icol == 'with_ARC' or icol == 'ideal_with_ARC' or icol =='ideal' or icol=='best':
            cols = sns.color_palette("husl", 4)
        else:
            cols = sns.cubehelix_palette(len(use_orders), start=.5, rot=-.9, reverse=True)

        GaAs = material('GaAs_WVASE')()
        Air = material('Air')()

        InAlP = material('InAlP_WVASE')()
        InGaP = material('InGaP_WVASE')()

        Ag = material('Ag_Jiang')()

        # MgF2 = material('MgF2')()
        # Ta2O5 = material('410', nk_db=True)()
        SiN = material('SiN_SE')()

        grating1 = [Layer(si(x[2] * x[1], 'nm'), SiN)]

        grating2 = [Layer(si((1 - x[1]) * x[2], 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),

                                                                       'radius': x[3], 'angle': 0}])]
        if icol == 'with_ARC' or icol == 'ideal_with_ARC':
            Al2O3 = material('354', nk_db=True)()
            solar_cell = SolarCell([Layer(material=Al2O3, width=si('70nm')),
                                       Layer(material=InGaP, width=si('19nm')),
                                    Layer(material=GaAs, width=si(x[0], 'nm')),
                                    Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                                   substrate=Ag)

            S4_setup = rcwa_structure(solar_cell, self.size, self.orders, self.options, Air, Ag)

            RAT = S4_setup.calculate()
            EQE_sim = 0.9 * 100 * (RAT['A_layer'][:, 1] + RAT['A_layer'][:, 2])
            Acomb = RAT['A_layer'][:, 2] + RAT['A_layer'][:, 1]
            Jsc = np.round(
                0.1 * (q / (h * c)) * np.trapz(S4_setup.wavelengths * 1e9 * Acomb * self.AM0,
                                               S4_setup.wavelengths * 1e9) / 1e9, 1)
            print('Jsc with ARC ', Jsc)


        else:
            solar_cell = SolarCell([Layer(material=InGaP, width=si('19nm')),
                                    Layer(material=GaAs, width=si(x[0], 'nm')),
                                    Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                                   substrate=Ag)

            S4_setup = rcwa_structure(solar_cell, self.size, self.orders, self.options, Air, Ag)

            RAT = S4_setup.calculate()
            EQE_sim = 0.9*100*(RAT['A_layer'][:,0] + RAT['A_layer'][:,1])
            Acomb = RAT['A_layer'][:, 0] + RAT['A_layer'][:, 1]
            Jsc = np.round(
                0.1 * (q / (h * c)) * np.trapz(S4_setup.wavelengths * 1e9 * Acomb * self.AM0,
                                               S4_setup.wavelengths * 1e9) / 1e9, 1)
            print('Jsc without ARC ', Jsc)



        # plt.figure()
        if icol == 'ideal':
            plt.plot(S4_setup.wavelengths * 1e9, EQE_sim, color=cols[0], label='Fully etched (' + str(Jsc) +')')

        elif icol == 'with_ARC':

            plt.plot(S4_setup.wavelengths * 1e9, EQE_sim, '--', color=cols[1], label='Partly etched ith ARC (' + str(Jsc) +')')

        elif icol == 'ideal_with_ARC':
            plt.plot(S4_setup.wavelengths * 1e9, EQE_sim, '--', color=cols[2], label='Fully etched with ARC (' + str(Jsc) +')')

        elif icol=='best':
            plt.plot(S4_setup.wavelengths * 1e9, EQE_sim, color=cols[3],
                     label='Best fit (' + str(Jsc) + ')')

        else:
            plt.plot(S4_setup.wavelengths*1e9, EQE_sim, label='n = ' + str(self.orders), color=cols[icol])


        # plt.show()
        return S4_setup.wavelengths*1e9, self.EQE_meas

    def calc_A(self, x):
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()


        return [self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2]]



def plot_R(obj, x, icol):

    cols = sns.cubehelix_palette(len(use_orders), start=.5, rot=-.9, reverse=True)
    GaAs = material('GaAs_WVASE')()
    Air = material('Air')()

    InAlP = material('InAlP_WVASE')()
    InGaP = material('InGaP_WVASE')()

    Ag = material('Ag_Jiang')()

    # MgF2 = material('MgF2')()
    # Ta2O5 = material('410', nk_db=True)()
    SiN = material('SiN_SE')()

    R_data = np.loadtxt('Talbot_precursor_R.csv', delimiter=',')


    grating1 = [Layer(si(x[2] * x[1], 'nm'), SiN)]

    grating2 = [Layer(si((1 - x[1]) * x[2], 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                                   'radius': x[3], 'angle': 0}])]

    solar_cell = SolarCell([Layer(material=GaAs, width=si('25nm')),
                            Layer(material=InGaP, width=si('19nm')),
                            Layer(material=GaAs, width=si(x[0], 'nm')),
                            Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                           substrate=Ag)

    S4_setup = rcwa_structure(solar_cell, obj.size, obj.orders, obj.options, Air, Ag)

    RAT = S4_setup.calculate()

    total_A = np.sum(RAT['A_layer'], axis=1)

    # plt.figure()

    plt.plot(S4_setup.wavelengths * 1e9, RAT['R'], label=str(obj.orders), color=cols[icol])


    return S4_setup.wavelengths * 1e9, interp1d(R_data[:, 0], R_data[:, 1])(S4_setup.wavelengths * 1e9)


maxiters=10
fig=plt.figure(figsize=[6.7,5])

for i1, n_orders in enumerate(use_orders):
    DE_class = EQE_fit(n_orders)

    if calc:

        DE_obj = DE(DE_class.evaluate, bounds=[[80, 90], [0, 1], [80, 120], [100, 125]], maxiters=maxiters)
        # GaAs thickness, SiN only fraction, total dielectric thickness, radius of disks
        # Can't parallelize again; RCWA already computing over wl in parallel, so use DE and not PDE

        # solve, i.e. minimize the problem
        res = DE_obj.solve(show_progress=True)

        # PDE_obj.solve() returns 5 things:
        # res[0] is a list of the parameters which gave the minimized value
        # res[1] is that minimized value
        # res[2] is the evolution of the best population (the best population from each iteration
        # res[3] is the evolution of the minimized value, i.e. the fitness over each iteration
        # res[4] is the evolution of the mean fitness over the iterations

        np.save('res_thinGaAsEQE_DE_'+str(n_orders), res)


    else:
        res = np.load('res_thinGaAsEQE_DE_'+str(n_orders) + '.npy', allow_pickle=True)
        # plot the result at these best parameters


    best_pop = res[0]
    print(str(n_orders) +' orders, parameters for best result:', best_pop, 'residual', res[1])
    params[i1, :4] = best_pop
    params[i1, 4] = res[1]
    ploteqe = DE_class.plot(best_pop, i1)

    # best_pop_evo = res[2]
    # best_fitn_evo = res[3]
    # mean_fitn_evo = res[4]
    # final_fitness = res[1]
    #
    # # plot evolution of the fitness of the best population per iteration
    # fig =plt.figure(figsize=[6,5])
    # plt.plot(np.arange(1,maxiters+1), -best_fitn_evo, 'k', label='Best')
    # plt.plot(np.arange(1,maxiters+1), -mean_fitn_evo, 'k--', label='Mean')
    # plt.xlabel('DE iteration')
    # plt.ylabel(r'$\mathrm{Fitness}~(J_{ph})$')
    #
    # plt.legend()
    # plt.title('b)', loc='left')
    # plt.show()


Jsc_EQE = np.round(0.1 * (q / (h * c)) * np.trapz(ploteqe[0] * ploteqe[1] * DE_class.AM0/100,
                                               ploteqe[0]) / 1e9, 1)
print('Jsc from integrating EQE ', Jsc_EQE)

plt.plot(*ploteqe, '--k', label='Measured EQE')
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE/Absorption (%)')

plt.xlim(300, 1000)
plt.ylim(0, 90)
plt.legend()
plt.title('a)', loc='left')
fig.savefig('thinGaAs_EQEDEoptim.pdf', bbox_inches='tight')
plt.show()


res = np.load('res_thinGaAsEQE_DE_'+str(85) + '.npy', allow_pickle=True)

fig=plt.figure(figsize=[6.7,5])
DE_class = EQE_fit(85)
ideal_pop = [86, 0, 110, 115]
ideal_pop[1] = 0
DE_class.plot(res[0], 'best')
DE_class.plot(ideal_pop, 'ideal')
DE_class.plot([85, 0.69, 110, 108], 'with_ARC')
DE_class.plot(ideal_pop, 'ideal_with_ARC')
plt.xlabel('Wavelength (nm)')
plt.ylabel('EQE/Absorption (%)')
plt.legend()
plt.xlim(300, 1000)
plt.ylim(0, 90)
plt.title('b)', loc='left')
fig.savefig('thinGaAs_EQEcomp.pdf', bbox_inches='tight')
plt.show()
#
# plt.figure()
#
#
# for i1, n_orders in enumerate(use_orders):
#     DE_class = EQE_fit(n_orders)
#     res = np.load('res_thinGaAsEQE_DE_' + str(n_orders) + '.npy', allow_pickle=True)
#     Rdata = plot_R(DE_class, res[0], i1)
#
# plt.legend()
# plt.plot(*Rdata, '--k', label='R measurement')
# plt.xlim(300, 1000)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Reflection (%)')
# plt.show()


# # TMM simulation
#
# GaAs = material('GaAs_WVASE')()
# Air = material('Air')()
#
# InAlP = material('InAlP_WVASE')()
# InGaP = material('InGaP_WVASE')()
#
# Ag = material('Ag_Jiang')()
#
# # MgF2 = material('MgF2')()
# # Ta2O5 = material('410', nk_db=True)()
# SiN = material('SiN_SE')()
#
# grating1 = [Layer(si(30, 'nm'), SiN)]
#
# wavelengths = np.linspace(303, 1000, 160) * 1e-9
#
# from solcore.absorption_calculator import OptiStack, calculate_rat
#
# solar_cell = SolarCell([Layer(material=InGaP, width=si('19nm')),
#                         Layer(material=GaAs, width=si(89, 'nm')),
#                         Layer(material=InAlP, width=si('18nm'))] + grating1, substrate=Ag)
#
#
# OS_layers = OptiStack(solar_cell, substrate=Ag, no_back_reflection=False)
#
# TMM_res = calculate_rat(OS_layers, wavelength=wavelengths*1e9,
#                         no_back_reflection=False)
#
# plt.figure()
# plt.plot(wavelengths*1e9, TMM_res['A_per_layer'][1] + TMM_res['A_per_layer'][2])
# plt.show()
#
#
