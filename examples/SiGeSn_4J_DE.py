# try optimizing an anti-reflection coating

import numpy as np

from solcore import material

import matplotlib.pyplot as plt

from solcore.optics.tmm import OptiStack
from solcore.optics.tmm import calculate_rat

# Import the DE implementations
from optimization.de import PDE
from solcore.light_source import LightSource

from solcore.constants import q

# Optimizing a four-junction cell using SiGeSn as the second-lowest bandgap material.
# Using a purely optical TMM simulation to calculate the photogenerated current in each sub-cell. The thing to
# optimize is then the current of the current-limiting cell in the structure; in other words we want to maximize the
# minimum sub-cell current. Since differential evolution does a minimization, we are actually minimizing the negative of
# this value.



# To use yabox for the DE, we need to define a class which sets up the problem and has an 'evaluate' function, which
# will actually calculate the value we are trying to minimize for each set of parameters.

class calc_min_Jsc():

    def __init__(self):


        T = 298
        wl = np.linspace(300, 1900, 800)

        # materials
        SiGeSn = material('SiGeSn')(T=T)

        GaAs = material('GaAs')(T=T)
        InGaP = material('GaInP')(In=0.5, T=T)
        Ge = material('Ge_WVASE')(T=T)

        # make these attributes of 'self' so they can be accessed by the class object
        # here I am also creating lists of wavelengths and corresponding n and k data from
        # the Solcore materials - the reason for this is that there is currently an issue with using the Solcore
        # material class in parallel computations. Thus the information for the n and k data is saved here as a list
        # rather than a material object.

        self.wl = wl
        self.SiGeSn = [self.wl, SiGeSn.n(self.wl*1e-9), SiGeSn.k(self.wl*1e-9)]
        self.Ge = [self.wl, Ge.n(self.wl*1e-9), Ge.k(self.wl*1e-9)]

        self.InGaP = [self.wl, InGaP.n(self.wl*1e-9), InGaP.k(self.wl*1e-9)]
        self.GaAs = [self.wl, GaAs.n(self.wl*1e-9), GaAs.k(self.wl*1e-9)]
        self.MgF2 = [self.wl, material('MgF2')().n(self.wl*1e-9), material('MgF2')().k(self.wl*1e-9)]
        self.Ta2O5 = [self.wl, material('410',
                                        nk_db=True)().n(self.wl*1e-9), material('410',
                                                                                nk_db=True)().k(self.wl*1e-9)]

        # assuming an AM1.5G spectrum
        self.spectr = LightSource(source_type='standard', version='AM1.5g', x=self.wl,
                           output_units='photon_flux_per_nm', concentration=1).spectrum(self.wl)[1]



    def evaluate(self, x):
        # create a list of layers with the format [thickness, wavelengths, n_data, k_data] for each layer.
        # This is one of the acceptable formats in which OptiStack can take information (look at the Solcore documentation
        # or at the OptiStack code for more info

        SC = [[x[0]] + self.MgF2, [x[1]] + self.Ta2O5, [x[2]] + self.InGaP, [x[3]] + self.GaAs, [x[4]] + self.SiGeSn, [300e3]+self.Ge]#, [x[5]] + self.Ge]

        # create the OptiStack. We set no_back_reflection to False because we DO want to include reflection at the back surface
        # (thin-film interference due to reflection at the SiGeSn/Ge interface)
        full_stack = OptiStack(SC, no_back_reflection=False)

        # calculate reflection, transmission, and absorption in each layer. We are specifying that the last layer,
        # a very thick Ge substrate, should be treated incoherently, otherwise we would see very narrow, unphysical oscillations
        # in the R/A/T spectra.
        RAT = calculate_rat(full_stack, self.wl, no_back_reflection=False, coherent=False,
                            coherency_list=['c', 'c', 'c', 'c', 'c', 'i'])

        # extract absorption per layer
        A_InGaP = RAT['A_per_layer'][3]
        A_GaAs = RAT['A_per_layer'][4]
        A_SiGeSn = RAT['A_per_layer'][5]
        A_Ge = RAT['A_per_layer'][6]

        ## calculate currents using the AM1.5 G spectrum for each layer
        Jsc_InGaP = 0.1 * q * np.trapz(A_InGaP* self.spectr, self.wl)
        Jsc_GaAs = 0.1 * q * np.trapz(A_GaAs * self.spectr, self.wl)
        Jsc_SiGeSn = 0.1 * q * np.trapz(A_SiGeSn* self.spectr, self.wl)
        Jsc_Ge = 0.1 * q* np.trapz(A_Ge * self.spectr, self.wl)

        # find the limiting current by checking which junction has the lowest current. Then take the negative since
        # we need to minimize (not maximize)
        limiting_Jsc = -min([Jsc_InGaP, Jsc_GaAs, Jsc_SiGeSn, Jsc_Ge])

        return limiting_Jsc

    def plot(self, x):
        # x is a list with all the layer thicknesses, except for the Ge (which is kept as a thick substrate)

        # this does basically what evaluate() does, but plots the results
        SC = [[x[0]] + self.MgF2, [x[1]] + self.Ta2O5, [x[2]] + self.InGaP,
              [x[3]] + self.GaAs, [x[4]] + self.SiGeSn, [300e3] + self.Ge]

        full_stack = OptiStack(SC, no_back_reflection=False)

        RAT = calculate_rat(full_stack, self.wl, no_back_reflection=False,
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



# number of iterations for Differential Evolution
maxiters=200

# class the DE algorithm is going to use, as defined above
DE_class = calc_min_Jsc()

# pass the function which will be minimized to the PDE (parallel differential evolution) solver. PDE calculates the
# results for each population in parallel to speed up the overall process. The bounds argument sets upper and lower bounds
# for each parameter.
PDE_obj = PDE(DE_class.evaluate, bounds=[[10,150], [10,105], [200, 1000], [500, 10000], [500, 10000]], maxiters=maxiters)

# solve, i.e. minimize the problem
res = PDE_obj.solve()

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


