# try optimizing an anti-reflection coating

import numpy as np

from solcore import si, material
from solcore.structure import Layer

import matplotlib.pyplot as plt

from solcore.optics.tmm import OptiStack
from solcore.optics.tmm import calculate_rat

# Import the DE implementations
from yabox.algorithms import DE, PDE
from solcore.light_source import LightSource

from solcore.constants import q, h, c
from solcore.interpolate import interp1d



#from solcore.material_system import create_new_material
#create_new_material('SiGeSn', 'SiGeSn_29111_n.txt', 'SiGeSn_29111_k.txt')
#create_new_material('SiGeSn_low', 'SiGeSn_29109_n.txt', 'SiGeSn_29109_k.txt')

class calc_R_diff():

    def __init__(self):


        T = 298
        wl = np.linspace(300, 1900, 800)
     
        SiGeSn = material('SiGeSn')(T=T)

        GaAs = material('GaAs')(T=T)
        InGaP = material('GaInP')(In=0.5, T=T)
        Ge = material('Ge_WVASE')(T=T)
        self.wl = wl
        self.SiGeSn = [self.wl, SiGeSn.n(self.wl*1e-9), SiGeSn.k(self.wl*1e-9)]
        self.Ge = [self.wl, Ge.n(self.wl*1e-9), Ge.k(self.wl*1e-9)]

        self.InGaP = [self.wl, InGaP.n(self.wl*1e-9), InGaP.k(self.wl*1e-9)]
        self.GaAs = [self.wl, GaAs.n(self.wl*1e-9), GaAs.k(self.wl*1e-9)]
        self.MgF2 = [self.wl, material('MgF2')().n(self.wl*1e-9), material('MgF2')().k(self.wl*1e-9)]
        self.Ta2O5 = [self.wl, material('410',
                                        nk_db=True)().n(self.wl*1e-9), material('410',
                                                                                nk_db=True)().k(self.wl*1e-9)]

        self.spectr = LightSource(source_type='standard', version='AM1.5g', x=self.wl,
                           output_units='photon_flux_per_nm', concentration=1).spectrum(self.wl)[1]



    def evaluate(self, x):

        SC = [[x[0]] + self.MgF2, [x[1]] + self.Ta2O5, [x[2]] + self.InGaP, [x[3]] + self.GaAs, [x[4]] + self.SiGeSn, [300e3]+self.Ge]#, [x[5]] + self.Ge]

        full_stack = OptiStack(SC, no_back_reflexion=False)
        #start=time()
        RAT = calculate_rat(full_stack, self.wl, no_back_reflexion=False, coherent=False,
                            coherency_list=['c', 'c', 'c', 'c', 'c', 'i'])
        #print(time()-start)
        A_InGaP = RAT['A_per_layer'][3]
        A_GaAs = RAT['A_per_layer'][4]
        A_SiGeSn = RAT['A_per_layer'][5]
        A_Ge = RAT['A_per_layer'][6]

        ## calculate currents
        Jsc_InGaP = 0.1 * q * np.trapz(A_InGaP* self.spectr, self.wl)
        Jsc_GaAs = 0.1 * q * np.trapz(A_GaAs * self.spectr, self.wl)
        Jsc_SiGeSn = 0.1 * q * np.trapz(A_SiGeSn* self.spectr, self.wl)
        Jsc_Ge = 0.1 * q* np.trapz(A_Ge * self.spectr, self.wl)
        #print('thick', x)
        #print([Jsc_InGaP, Jsc_GaAs, Jsc_SiGeSn, Jsc_Ge])
        max_Jsc = -min([Jsc_InGaP, Jsc_GaAs, Jsc_SiGeSn, Jsc_Ge])

        return max_Jsc

    def plot(self, x):
        SC = [[x[0]] + self.MgF2, [x[1]] + self.Ta2O5, [x[2]] + self.InGaP,
              [x[3]] + self.GaAs, [x[4]] + self.SiGeSn, [1e6] + self.Ge]

        full_stack = OptiStack(SC, no_back_reflexion=False)

        RAT = calculate_rat(full_stack, self.wl, no_back_reflexion=False,
                            coherent=False, coherency_list=['c', 'c', 'c', 'c', 'c', 'i'])

        A_InGaP = RAT['A_per_layer'][3]
        A_GaAs = RAT['A_per_layer'][4]
        A_SiGeSn = RAT['A_per_layer'][5]
        A_Ge = RAT['A_per_layer'][6]

        plt.figure()
        plt.plot(self.wl, A_InGaP, label='InGaP')
        plt.plot(self.wl, A_GaAs, label='A_GaAs')
        plt.plot(self.wl, A_SiGeSn, label='SiGeSn')
        plt.plot(self.wl, A_Ge, label = 'Ge')
        plt.plot(self.wl, RAT['T'], label='T')
        plt.plot(self.wl, RAT['R'], label='R')
        plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.show()

from time import time
maxiters=2000
a = calc_R_diff()
start=time()
DE_obj = PDE(a.evaluate, bounds=[[10,150], [10,105], [200, 1000], [500, 10000], [500, 10000]], maxiters=maxiters)
# solve returns weird reshaped array?

res = DE_obj.solve()

best_pop = res[0].diagonal()
print(time()-start)
print(best_pop, res[1])
a.plot(best_pop)


import seaborn as sns
from cycler import cycler
best_pop_evo = res[2]
best_fitn_evo = np.array(res[3])
mean_fitn_evo = res[4]
final_fitness = res[1]
pal = sns.cubehelix_palette(maxiters+1, start=.5, rot=-.9)


cols = cycler('color', pal)

plt.rcParams['axes.prop_cycle'] = cols

plt.figure()
plt.plot(-best_fitn_evo, '-k')
plt.xlabel('iteration')
plt.ylabel('fitness')
plt.show()

plt.figure()
plt.plot(best_pop_evo[:,3])
plt.show()
# it = DE_obj.iterator()
#
# times = []
# fitness = []
# start = time()
# i = 0
# for status in it:
#     curr = time()-start
#     times.append(curr)
#     fitness.append(status.best_fitness)
#     i += 1
#     if i > maxiters: #or curr > stop_after:
#         break
#
# plt.figure()
# plt.plot(times, fitness)
# plt.show()
#
# plt.figure()
# plt.plot(np.log(-np.array(fitness)))
# plt.show()
#
# a.plot()