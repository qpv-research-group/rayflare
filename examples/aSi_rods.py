from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
from solcore.constants import q
import seaborn as sns
from matplotlib import rc
from joblib import Parallel, delayed

# Paper: https://doi.org/10.1016/j.solmat.2016.09.005

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 15})

pal = sns.color_palette("husl", 3)


wavelengths = np.array([4e-6])

# x = 2000

# lattice vectors for the grating. Units are in nm!
options = default_options()
options.wavelength = wavelengths
options.orders = 60

rad_list = np.linspace(100, 1000, 40)
lattice_constant = np.linspace(1000, 3000, 40)


def fixed_lattice_constant(x, rad_list):
    MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
    Air = material("Air")()
    aSi = material("Si")()
    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    T_result = np.zeros(len(rad_list))

    for j1, rad in enumerate(rad_list):

        grating_circles = [{"type": "circle", "mat": aSi, "center": (0, 0), "radius": rad}]

        layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]

        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=MgF2, transmission=Air)
        RAT = S4_setup.calculate(options)
        T_result[j1] = RAT["T"][0]

    return T_result


T_result = Parallel(n_jobs=-1)(delayed(fixed_lattice_constant)(x, rad_list) for x in lattice_constant)

T_result = np.array(T_result)
    # T_result = np.zeros((len(lattice_constant), len(rad_list)))
    #
    # for i1, x in enumerate(lattice_constant):
    #     print(i1)
    #     size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))
    #
    #     for j1, rad in enumerate(rad_list):
    #
    #         grating_circles = [{"type": "circle", "mat": aSi, "center": (0, 0), "radius": rad}]
    #
    #         layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]
    #
    #         S4_setup = rcwa_structure(layers, size=size, options=options, incidence=MgF2, transmission=Air)
    #         RAT = S4_setup.calculate(options)
    #         T_result[i1, j1] = RAT["T"][0]



plt.figure()
plt.imshow(T_result, aspect="auto", extent=[rad_list[0], rad_list[-1], lattice_constant[0], lattice_constant[-1]],
           origin="lower")
plt.colorbar()
plt.xlabel("Radius (nm)")
plt.ylabel("Lattice constant (nm)")
plt.title("Transmission")
plt.show()
