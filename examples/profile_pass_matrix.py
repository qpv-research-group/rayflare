import numpy as np

from rayflare.textures import regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.options import default_options
from rayflare.angles import make_angle_vector
from rayflare.utilities import make_absorption_function


from solcore import material, si
from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver
from solcore.light_source import LightSource
from solcore.constants import q

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# GaAs/GaAs/Si solar cell

# ARC

# Electron barrier (GaInP)
# Absorber (GaAs, p on n)
# Hole barrier (GaInP)
# Tunnel - n-GaAs?
# Hole barrier (InAlP)
# Absorber (GaAs, n on p)
# Electron barrier (GaInP)
# n-GaAs bonding
# n-Si
# p-Si
# rear surface:


wavelengths = np.linspace(250, 1200, 100)*1e-9

options = default_options()
options.wavelengths = wavelengths
options.project_name = 'GaAs_GaAs_Si'
options.n_theta_bins = 100
options.nx = 5
options.ny = 5
options.depth_spacing = si('1nm')
options.depth_spacing_bulk = si('100nm')
_, _, angle_vector = make_angle_vector(options['n_theta_bins'], options['phi_symmetry'],
                                       options['c_azimuth'])
options.bulk_profile = True
options.n_rays = options.nx**2*int(len(angle_vector)/2)

Air = material('Air')()
Al2O3 = material("Al2O3")()
Ag = material("Ag_Jiang")()
aSi_i = material("aSi_i")()
aSi_p = material("aSi_p")()
aSi_n = material("aSi_n")()
ITO_back = material("ITO_back")()
GaAs = material("GaAs")()
InAlP = material("AlInP")(Al=0.5)
GaInP = material("GaInP")(In=0.5)
Si = material("Si")()
MgF2 = material("203", nk_db=True)()
Ta2O5 = material("410", nk_db=True)()

GaAs_1_th = 120e-9
GaAs_2_th = 1200e-9

front_materials = [Layer(50e-9,  MgF2), Layer(40e-9, Ta2O5),
                   Layer(30e-9, GaInP), Layer(GaAs_1_th, GaAs), Layer(30e-9, InAlP),
                   Layer(20e-9, GaAs),
                   Layer(30e-9, GaInP), Layer(GaAs_2_th, GaAs), Layer(30e-9, InAlP),
                   Layer(100e-9, GaAs),
                   Layer(6.5e-9, aSi_p), Layer(6.5e-9, aSi_i)]
back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_n), Layer(240e-9, ITO_back)]


surf_back = regular_pyramids(elevation_angle=55, upright=False)

front_surf = Interface('TMM', layers=front_materials, name='GaAs_GaAs',
                       coherent=True, prof_layers=[4, 8])
back_surf = Interface('RT_TMM', texture=surf_back, layers=back_materials,
                      name='Si_HIT_rear', prof_layers=[1,3],
                      coherent=True)

bulk_Si = BulkLayer(250e-6, Si, name = 'Si_bulk') # bulk thickness in m

SC = Structure([front_surf, bulk_Si, back_surf], incidence=Air, transmission=Ag)

process_structure(SC, options)

results = calculate_RAT(SC, options)


RAT = results[0]
results_per_pass = results[1]

# only select absorbing layers, sum over passes
results_per_layer_front = np.sum(results_per_pass['a'][0], 0)

results_per_layer_back = np.sum(results_per_pass['a'][1], 0)

allres = np.flip(np.vstack((RAT['R'],
                           results_per_layer_front[:,3].T,
                            results_per_layer_front[:,7].T,
                            RAT['A_bulk'],
                            RAT['T'])), 0)


spectr_flux = LightSource(source_type='standard', version='AM1.5g', x=wavelengths,
                           output_units='photon_flux_per_m', concentration=1).spectrum(wavelengths)[1]

Jph_GaAs_1 = q * np.trapz(results_per_layer_front[:,3] * spectr_flux, wavelengths)/10 # mA/cm2
Jph_GaAs_2 = q * np.trapz(results_per_layer_front[:,7] * spectr_flux, wavelengths)/10 # mA/cm2
Jph_Si = q * np.trapz(RAT['A_bulk'][0] * spectr_flux, wavelengths)/10 # mA/cm2


pal = sns.cubehelix_palette(allres.shape[0], start=.5, rot=-.9)
pal.reverse()
cols = cycler('color', pal)
params = {'axes.prop_cycle': cols}

plt.rcParams.update(params)
# plot total R, A, T
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
ax.plot(options['wavelengths']*1e6,
             allres.T)
ax.set_xlabel(r'Wavelength ($\mu$m)')
ax.set_ylabel('Absorption/Emissivity')
ax.set_ylim(0, 1)
plt.legend(labels=['Ag', 'Bulk Si', 'GaAs 2', 'GaAs 1', 'R'])
plt.show()

print(Jph_GaAs_1, Jph_GaAs_2, Jph_Si)

profile_front = results[2][0]
profile_Si = results[3][0]
profile_back = results[2][1]

positions, absorb_fn = make_absorption_function([profile_front, profile_Si, profile_back], SC, options, True)

external_R = RAT['R'][0, :]

Si_SC = material("Si")
GaAs_SC = material("GaAs")
T = 300

p_material_Si = Si_SC(T=T, Na=si(1e21, "cm-3"), electron_diffusion_length=si("10um"),
                hole_mobility=50e-4, relative_permittivity=11.68)
n_material_Si = Si_SC(T=T, Nd=si(1e16, "cm-3"), hole_diffusion_length=si("290um"),
                electron_mobility=400e-4, relative_permittivity=11.68)


p_material_GaAs = GaAs_SC(T=T, Na=si(3e18, "cm-3"), electron_diffusion_length=si("400nm"),
                       hole_mobility=50e-4, relative_permittivity=12.4)
n_material_GaAs = GaAs_SC(T=T, Nd=si(1e18, "cm-3"), hole_diffusion_length=si("1um"),
                       electron_mobility=100e-4, relative_permittivity = 12.4)

from solcore.solar_cell_solver import default_options as defaults_solcore

options_sc = defaults_solcore
options_sc.optics_method = "external"
options_sc.position = positions
options_sc.light_iv = True
options_sc.wavelength = wavelengths
options_sc.mpp = True
options_sc.theta = options.theta_in*180/np.pi
V = np.linspace(0, 2.5, 250)
options_sc.voltages = V

solar_cell = SolarCell([Layer(50e-9,  MgF2), Layer(40e-9, Ta2O5),
                   Layer(30e-9, GaInP),
                   Junction([Layer(GaAs_1_th/2, p_material_GaAs, role="emitter"),
                             Layer(GaAs_1_th/2, n_material_GaAs, role="base")], kind="DA"),
                   Layer(30e-9, InAlP),
                   Layer(20e-9, GaAs),
                   Layer(30e-9, GaInP),
                   Junction([Layer(150e-9, p_material_GaAs, role="emitter"),
                             Layer(GaAs_2_th-150e-9, n_material_GaAs, role="base")], kind="DA"),
                   Layer(30e-9, InAlP),
                   Layer(100e-9, GaAs),
                   Layer(6.5e-9, aSi_p), Layer(6.5e-9, aSi_i),
                   Junction([Layer(500e-9, p_material_Si, role="emitter"),
                             Layer(250e-6-500e-9, n_material_Si, role="base")], kind="DA"),
                   Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_n),
                   Layer(240e-9, ITO_back)],
                  external_reflected = external_R,
                                       external_absorbed = absorb_fn)



solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(options['wavelengths']*1e9, allres.T, color='grey')
plt.plot(wavelengths*1e9, solar_cell.absorbed, 'k--', label='Absorbed (integrated)')
plt.plot(wavelengths*1e9, solar_cell[3].eqe(wavelengths), 'r-', label='GaAs EQE')
plt.plot(wavelengths*1e9, solar_cell[7].eqe(wavelengths), 'b-', label='Si EQE')
plt.plot(wavelengths*1e9, solar_cell[12].eqe(wavelengths), 'g-', label='Si EQE')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()


plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], '--k', label='Total')
plt.plot(V, -solar_cell[3].iv(V), 'r', label='GaAs (1)')
plt.plot(V, -solar_cell[7].iv(V), 'b', label='GaAs (2)')
plt.plot(V, -solar_cell[12].iv(V), 'g', label='Si')
plt.ylim(-20, 250)
plt.xlim(0, 2.5)
plt.legend()
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)') #The expected values of Isc and Voc are 372 A/m^2 and 0.63 V respectively
plt.show()