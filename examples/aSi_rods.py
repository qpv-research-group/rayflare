from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from time import time

# Paper: https://doi.org/10.1002/adom.201700585

# Define materials:
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
Air = material("Air")()
aSi = material("Si")() # the paper uses a-Si:H, but the refractive index of the database Si here
# is very similar at 5um (n = 3.5)

# 5 um
wavelengths = np.array([4e-6])

options = default_options()
options.wavelength = wavelengths
options.orders = 43 # number of diffraction orders. Note that due to the truncation scheme,
# the actual number of orders used can be slightly higher or lower than this. A higher number should
# give a more accurate result, since the Fourier transformed representation is more accurate, but
# takes longer to calculate.
options.A_per_order = True # this makes the RCWA implementation (S4) calculate the power per diffraction order
# and field amplitudes, which are required to calculate the phase. By default, if this option is not turned on,
# only the total R and T will be calculated.
options.pol = 's' # polarization of the incident plane wave. This can be 's', 'p', 'u' (equal mixture of s and p),
# or a Jones vector such as options.pol = (1/np.sqrt(2), 1j/np.sqrt(2)) (circular polarization)

rad_list = np.linspace(100, 1000, 41) # list of pillar radii to scan through

lattice_constant = np.linspace(1000, 3000, 61)
# list of lattice constants to scan through (distance between centre of pillars)

# 21 radii, 31 lattice constants: miniconda python takes 101 s
# "normal" python installed via pyenv takes 43 s?


# Here we define a function which calculates the results for all the different radii at some
# specific lattice constant. Will use this below to execute in parallel.
def fixed_lattice_constant(x, rad_list):

    # Unit cell vectors (real space) for the hexagonal layout. The unit cell is a triangle.
    size = ((x, 0), (x / 2, np.sin(np.pi / 3) * x))

    # Empty arrays to store results
    T_result = np.zeros(len(rad_list))
    T_amp = np.zeros(len(rad_list), dtype=complex)
    inc_amp = np.zeros(len(rad_list), dtype=complex)

    # Loop through radii
    for j1, rad in enumerate(rad_list):

        # Add the pillar with the desired radius to the layer; background material of the layer is air
        grating_circles = [{"type": "circle", "mat": aSi, "center": (0, 0), "radius": rad}]
        layers = [Layer(material=Air, width=si("2000nm"), geometry=grating_circles)]

        # Make the object to do RCWA calculations. Assume light is incident from the MgF2 which is below
        # the pillars in the paper.
        S4_setup = rcwa_structure(layers, size=size, options=options, incidence=MgF2, transmission=Air)
        RAT = S4_setup.calculate(options)

        # Store the results; we care about total transmission, and want the amplitudes of the incident
        # light and the transmitted light (first entry in the last is always (0, 0) to calculate the
        # phase change.
        T_result[j1] = RAT["T"][0]
        T_amp[j1] = RAT["T_amplitudes"][0][0][0]
        inc_amp[j1] = RAT['R_amplitudes'][0][0][0]

    return T_result, T_amp, inc_amp

# Run in parallel on all available CPUs (can change n_jobs to a specific number, -1 uses all available)
# looping through all the lattice constants

start = time()
allres = Parallel(n_jobs=-1, prefer="threads")(delayed(fixed_lattice_constant)(x, rad_list) for x in lattice_constant)
print("Time taken: ", time() - start)

T_result = np.stack([item[0] for item in allres])
T_amp = np.stack([item[1] for item in allres])
inc_amp = np.stack([item[2] for item in allres])

# calculate phase and put it in the range 0, 2pi:
phase_res = np.angle(T_amp) # this is in range -pi to pi
phase_res[phase_res < 0] = 2*np.pi - np.abs(phase_res[phase_res < 0])


plt.figure()
plt.imshow(phase_res/np.pi, aspect="auto",
           extent=[rad_list[0], rad_list[-1], lattice_constant[0], lattice_constant[-1]],
           origin="lower", cmap="Spectral_r")
plt.colorbar()
plt.xlabel("Radius (nm)")
plt.ylabel("Lattice constant (nm)")
plt.title(r"Phase (units of $\pi$)")
plt.show()

max_radius = 610
# pick out results for a lattice constant of 2000 nm (fig 1d in paper):

T_result_2000 = T_result[np.argmin(np.abs(lattice_constant - 2000))]
T_amp_result_2000 = phase_res[np.argmin(np.abs(lattice_constant - 2000))]
phase_result_2000 = T_amp_result_2000[rad_list < max_radius]

fig, ax = plt.subplots()
ax.plot(rad_list, T_result_2000, label="Transmission")
ax.set_xlabel("Radius (nm)")
ax.set_ylabel("Transmission")
ax2 = ax.twinx()
ax2.plot(rad_list[rad_list < max_radius], phase_result_2000 / np.pi, label="Phase", color="r")
ax2.set_ylabel("Phase (units of $\pi$)")
ax.set_xlim(100, max_radius)
ax.legend()
ax2.legend()
plt.show()

# Plot the (x, y) E field amplitude in transmission.
# colors for each point:
colors = sns.color_palette("viridis", len(rad_list[rad_list < max_radius]))
T_amp_result_2000 = T_amp[np.argmin(np.abs(lattice_constant - 2000))]
T_amp_result_2000 = T_amp_result_2000[rad_list < max_radius]

fig, ax = plt.subplots()
for j, rad in enumerate(rad_list[rad_list < max_radius]):
    if j % 5 == 0:
        ax.plot(T_amp_result_2000[j].real, T_amp_result_2000[j].imag, 'o', color=colors[j], label=rad)
    else:
        ax.plot(T_amp_result_2000[j].real, T_amp_result_2000[j].imag, 'o', color=colors[j])
ax.plot(T_amp_result_2000.real, T_amp_result_2000.imag, 'k', alpha=0.5)

# show axes at x = 0 and y = 0:
ax.grid(True)
sns.despine(ax=ax, offset=0) # the important part here
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.set_xlabel('Re{E}')
ax.set_ylabel('Im{E}')
ax.xaxis.set_label_coords(1.01, 0.48)
ax.yaxis.set_label_coords(0.48, 1.03)
plt.legend()
plt.show()



