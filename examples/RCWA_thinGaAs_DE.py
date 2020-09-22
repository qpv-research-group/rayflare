import numpy as np
from solcore import si, material
from solcore.structure import Layer
from rigorous_coupled_wave_analysis.rcwa import rcwa_structure

import matplotlib.pyplot as plt

# Import the DE implementations
from solcore.optimization import PDE, DE

from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell
from cycler import cycler
import seaborn as sns

wavelengths = np.linspace(250, 930, 80) * 1e-9
EQE_wl = np.linspace(250.1, 929.9, 700)


pal = sns.cubehelix_palette(6, start=.5, rot=-.9, reverse=True)

cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols
plt.rcParams['font.size'] = 11

calc = False


# define the problem
class Jsc_optim:
    def __init__(self):
        GaAs = material('GaAs_WVASE')()
        Air = material('Air')()

        InAlP = material('InAlP_WVASE')()
        InGaP = material('InGaP_WVASE')()

        Ag = material('Ag_Jiang')()

        SiN = material('SiN_SE')()


        x = 600

        # anti-reflection coating
        ARC1 = [Layer(si('60nm'), SiN)]
        size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

        wavelengths = np.linspace(250, 930, 200) * 1e-9


        RCWA_wl = wavelengths

        options = {'depth_spacing': 0.5,
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

        grating = [Layer(si(100, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                        'radius': 100, 'angle': 0}])]
        solar_cell = SolarCell(ARC1 + [Layer(material=InGaP, width=si('19nm')),
                                       Layer(material=GaAs, width=si('86nm')),
                                       Layer(material=InAlP, width=si('18nm'))] + grating)
        self.S4_setup = rcwa_structure(solar_cell, size, 37, options, Air, Ag)
        spect = np.loadtxt('AM0.csv', delimiter=',')

        self.AM0 = interp1d(spect[:, 0], spect[:, 1])(wavelengths*1e9)


    def evaluate(self, x):
        # x[0] : ARC thickness
        # x[1]: grating thickness
        # x[2]: disk radius
        # x[3]: period
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()
        Jsc = 0.1 * (q / (h * c))* np.trapz(self.S4_setup.wavelengths*1e9*RAT['A_layer'][:, 2] * self.AM0,
                                            self.S4_setup.wavelengths*1e9)/1e9

        return -Jsc

    def plot(self, x):
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()

        Jsc_ARC = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['A_layer'][:, 0] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)
        Jsc_InGaP = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['A_layer'][:, 1] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)
        Jsc = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['A_layer'][:, 2] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)
        Jsc_InAlP = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['A_layer'][:, 3] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)
        Jsc_grating = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['A_layer'][:, 4] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)
        Jsc_R = np.round(0.1 * (q / (h * c)) * np.trapz(self.S4_setup.wavelengths * 1e9 * RAT['R'] * self.AM0,
                                             self.S4_setup.wavelengths * 1e9) / 1e9, 1)

        fig =plt.figure(figsize=[6,5])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2], label= 'GaAs p-n (' + str(Jsc) + ')')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['R'], '--', label= 'Reflection (' + str(Jsc_R) + ')')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,4], label= 'Grating (' + str(Jsc_grating) + ')')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,1], label= 'InGaP (' + str(Jsc_InGaP) + ')')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,0], label= 'ARC (' + str(Jsc_ARC) + ')')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,3], label= 'InAlP (' + str(Jsc_InAlP) + ')')

        plt.ylim([0, 0.8])
        plt.legend(loc='upper left')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Absorbed/reflected fraction')
        plt.title('a)', loc='left')
        fig.savefig('thinGaAs_DEoptim.pdf', bbox_inches='tight')
        plt.show()


    def calc_A(self, x):
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()


        return [self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2]]


maxiters=50
DE_class= Jsc_optim()

if calc:


    DE_obj = DE(DE_class.evaluate, crossover=0.8, mutation=(0.2, 1), bounds=[[20, 150], [50,150], [0, 300], [200, 1000]], maxiters=maxiters)
    # solve returns weird reshaped array?

    # solve, i.e. minimize the problem
    res = DE_obj.solve(show_progress=True)

    # PDE_obj.solve() returns 5 things:
    # res[0] is a list of the parameters which gave the minimized value
    # res[1] is that minimized value
    # res[2] is the evolution of the best population (the best population from each iteration
    # res[3] is the evolution of the minimized value, i.e. the fitness over each iteration
    # res[4] is the evolution of the mean fitness over the iterations

    np.save('res_thinGaAs_DE', res)


else:
    res = np.load('res_thinGaAs_DE.npy', allow_pickle=True)
    # plot the result at these best parameters

best_pop = res[0]
print('parameters for best result:', best_pop, res[1])

DE_class.plot(best_pop)

best_pop_evo = res[2]
best_fitn_evo = res[3]
mean_fitn_evo = res[4]
final_fitness = res[1]

# plot evolution of the fitness of the best population per iteration
fig =plt.figure(figsize=[6,5])
plt.plot(np.arange(1,51), -best_fitn_evo, 'k', label='Best')
plt.plot(np.arange(1,51), -mean_fitn_evo, 'k--', label='Mean')
plt.xlabel('DE iteration')
plt.ylabel(r'$\mathrm{Fitness}~(J_{ph})$')
plt.xlim(1, 50)
plt.ylim(15, 24)
plt.legend()
plt.title('b)', loc='left')
fig.savefig('thinGaAs_DEoptim_iter.pdf', bbox_inches='tight')
plt.show()




