# try optimizing an anti-reflection coating

import numpy as np
import seaborn as sns
from cycler import cycler

from solcore import si, material
from solcore.structure import Layer
from rigorous_coupled_wave_analysis.rcwa import RCWA_optim

import matplotlib.pyplot as plt

# Import the DE implementations
from yabox.algorithms import DE


from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell



# define the problem
class R_fit:
    def __init__(self):
        GaAs = material('GaAs_WVASE')()
        Air = material('Air')()

        InAlP = material('InAlP_WVASE')()
        InGaP = material('InGaP_WVASE')()

        Ag = material('Ag_Jiang')()

        #MgF2 = material('MgF2')()
        #Ta2O5 = material('410', nk_db=True)()
        SiN = material('SiN_SE')()

        x = 500

        R_data = np.loadtxt('Talbot_precursor_R.csv', delimiter=',')



        # anti-reflection coating
        #ARC1 = [Layer(si('60nm'), SiN)]
        size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))


        wavelengths = np.linspace(303, 1000, 80) * 1e-9


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
        planar_SiN = [Layer(si('50nm'), SiN)]

        grating1 = [Layer(si(100, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Air, 'center': (0, 0),
                                                        'radius': 110, 'angle': 0}])]
        grating2 = [Layer(si(100, 'nm'), SiN, geometry=[{'type': 'circle', 'mat': Ag, 'center': (0, 0),
                                                        'radius': 110, 'angle': 0}])]
        solar_cell = SolarCell([Layer(material=GaAs, width=si('25nm')),
                                        Layer(material=InGaP, width=si('19nm')),
                                       Layer(material=GaAs, width=si('86nm')),
                                       Layer(material=InAlP, width=si('18nm'))] + grating1 + grating2,
                               substrate=Ag)
        self.S4_setup = RCWA_optim(solar_cell, size, 37, options, Air, Ag)
        spect = np.loadtxt('AM0.csv', delimiter=',')
        self.AM0 = interp1d(spect[:, 0], spect[:, 1])(wavelengths*1e9)
        self.R_meas = interp1d(R_data[:,0], R_data[:,1])(wavelengths*1e9)

    def evaluate(self, x):

        self.S4_setup.set_widths([[25, 19, 86, 18, x[0], x[1]]])

        RAT = self.S4_setup.calculate()
        R_sim = RAT['R']

        # least squares?
        residual = np.sum((R_sim-self.R_meas)**2)
        print(residual)

        return residual

    def plot(self, x):
        last_thick = 106 - x[0] - x[1]
        if last_thick < 0:
            last_thick = 0
        self.S4_setup.set_widths([[25, 19, 86, 18, x[0], x[1]]])

        RAT = self.S4_setup.calculate()

        plt.figure()
        plt.plot(self.S4_setup.wavelengths*1e9, RAT['R'], '-')
        plt.plot(self.S4_setup.wavelengths*1e9, self.R_meas, '--')

        plt.ylim([0, 1])

        plt.xlabel('Wavelength (nm)')
        plt.show()

    def calc_A(self, x):
        self.S4_setup.set_widths([[x[0], 19, 86, 18, x[1]]])
        self.S4_setup.edit_geom_list(5, 0, {'radius': x[2]})
        self.S4_setup.set_size(((x[3], 0), (x[3] / 2, np.sin(np.pi / 3) * x[3])))
        RAT = self.S4_setup.calculate()


        return [self.S4_setup.wavelengths*1e9, RAT['A_layer'][:,2]]


from time import time
maxiters=10
a = R_fit()

b = a.evaluate([10, 10, 10])

start=time()
DE_obj = DE(a.evaluate, crossover=0.8, mutation=(0.2, 1), bounds=[[0, 106], [0, 106]], maxiters=maxiters)
# solve returns weird reshaped array?

res = DE_obj.solve()

best_pop = res[0].diagonal()
print(time()-start)
print(best_pop, res[1])

pal = sns.cubehelix_palette(8, start=.5, rot=-.9)

cols = cycler('color', pal)
plt.rcParams['axes.prop_cycle'] = cols
a.plot(best_pop)

# best_pop_evo = res[2]
# best_fitn_evo = np.array(res[3])
# mean_fitn_evo = res[4]
# final_fitness = res[1]
#
#
# plt.figure()
# plt.plot(-best_fitn_evo)
# plt.xlabel('iteration')
# plt.ylabel('fitness')
# plt.show()
#
# unique_best_pop = np.unique(np.array(best_pop_evo), axis=0)
# n_unique_pop = unique_best_pop.shape[0]
#
# pal = sns.cubehelix_palette(n_unique_pop, start=.5, rot=-.9)
#
#
# cols = cycler('color', pal)
#
# plt.rcParams['axes.prop_cycle'] = cols
#
# plt.figure()
# for i1 in range(n_unique_pop):
#     plt.plot(*a.calc_A(unique_best_pop[i1, :]), label = str(i1))
#
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.show()
#
# plt.figure()
# plt.plot(best_pop_evo[:,3])
# plt.show()
#
# plt.figure()
# plt.plot(best_pop_evo[:,2]/best_pop_evo[:,3])
# plt.show()

