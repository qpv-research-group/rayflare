import numpy as np
from solcore import si, material
from solcore.structure import Layer
from rigorous_coupled_wave_analysis.rcwa import rcwa_structure

import matplotlib.pyplot as plt

# Import the DE implementations
from optimization.de import DE

from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell

wavelengths = np.linspace(250, 930, 80) * 1e-9
EQE_wl = np.linspace(250.1, 929.9, 700)


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

        wavelengths = np.linspace(250, 930, 80) * 1e-9


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

        plt.figure()
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['R'], '--')
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,0])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,1])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,3])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,4])
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['T'], '--')
        plt.ylim([0, 1])
        plt.legend(['R', 'SiN', 'InGaP', 'GaAs', 'InAlP', 'grating', 'T'])
        plt.xlabel('Wavelength (nm)')
        plt.show()

    def calc_A(self, x):
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()


        return [self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2]]


maxiters=50
DE_class= Jsc_optim()

DE_obj = DE(DE_class.evaluate, crossover=0.8, mutation=(0.2, 1), bounds=[[20, 150], [50,150], [0, 300], [200, 1000]], maxiters=maxiters)
# solve returns weird reshaped array?

# solve, i.e. minimize the problem
res = DE_obj.solve()

# PDE_obj.solve() returns 5 things:
# res[0] is a list of the parameters which gave the minimized value
# res[1] is that minimized value
# res[2] is the evolution of the best population (the best population from each iteration
# res[3] is the evolution of the minimized value, i.e. the fitness over each iteration
# res[4] is the evolution of the mean fitness over the iterations

# best population:
best_pop = res[0]

print('parameters for best result:', best_pop, res[1])

# plot the result at these best parameters
DE_class.plot(best_pop)

best_pop_evo = res[2]
best_fitn_evo = res[3]
mean_fitn_evo = res[4]
final_fitness = res[1]

# plot evolution of the fitness of the best population per iteration

plt.figure()
plt.plot(-best_fitn_evo, '-k')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.title('Best fitness')
plt.show()

plt.figure()
plt.plot(-mean_fitn_evo, '-k')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.title('Mean fitness')
plt.show()


