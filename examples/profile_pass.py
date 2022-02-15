import numpy as np

from rayflare.ray_tracing import rt_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.textures import regular_pyramids, planar_surface
from rayflare.options import default_options
from rayflare.utilities import make_absorption_function

from solcore import material, si
from solcore.solar_cell import SolarCell, Layer, Junction
from solcore.solar_cell_solver import solar_cell_solver

# imports for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# setting up some colours for plotting
pal = sns.color_palette("husl", 4)


# setting up Solcore materials
Air = material('Air')()
Si = material('Si')()
GaAs = material('GaAs')()


# number of x and y points to scan across
nxy = 25

wl = np.linspace(300, 1201, 50) * 1e-9

# setting options
options = default_options()
options.wavelengths = wl
options.nx = nxy
options.ny = nxy
options.n_rays = 2 * nxy ** 2
options.depth_spacing = si('5nm')
options.parallel = True
options.theta_in = 0.5

GaAs_total_d = si('3um')
Si_total_d = si('300um')

flat_surf = planar_surface(size=2) # pyramid size in microns
triangle_surf = regular_pyramids(55, upright=False, size=2)

# set up ray-tracing options
rtstr = rt_structure(textures=[triangle_surf, triangle_surf, triangle_surf],
                    materials = [GaAs, Si],
                    widths=[GaAs_total_d, Si_total_d], incidence=Air, transmission=Air)
result_rt = rtstr.calculate_profile(options)

total_A_rt = result_rt['A_per_layer']
total_R_rt = result_rt['R']
profile_rt = result_rt['profile']

options.coherent = False
options.coherency_list = ['c', 'i']

tmmstr = tmm_structure([Layer(GaAs_total_d, GaAs), Layer(Si_total_d, Si)], incidence=Air, transmission=Air)
result_tmm = tmmstr.calculate_profile(options)

total_A_tmm = result_tmm['A_per_layer']
total_R_tmm = result_tmm['R']
profile_tmm = result_tmm['profile']

rcwastr = rcwa_structure([Layer(GaAs_total_d, GaAs), Layer(Si_total_d, Si)], ((0, 100), (100, 0)),
                         options, incidence=Air, transmission=Air)
result_rcwa = rcwastr.calculate_profile(options)

total_A_rcwa = result_rcwa['A_per_layer']
total_R_rcwa = result_rcwa['R']
profile_rcwa = result_rcwa['profile']


Si_SC = material("Si")
GaAs_SC = material("GaAs")
T = 300

n_material_GaAs_width = si("300nm")
p_material_GaAs_width = rtstr.widths[0] - n_material_GaAs_width

n_material_GaAs = GaAs_SC(Nd=si(3e18, "cm-3"), hole_diffusion_length=si("400nm"),
                       electron_mobility=50e-4, relative_permittivity=12.4)
p_material_GaAs = GaAs_SC(Na=si(1e17, "cm-3"), electron_diffusion_length=si("1um"),
                       electron_mobility=100e-4, relative_permittivity = 12.4)

n_material_Si_width = si("500nm")
p_material_Si_width = rtstr.widths[1] - n_material_Si_width

n_material_Si = Si_SC(T=T, Nd=si(1e21, "cm-3"), hole_diffusion_length=si("10um"),
                electron_mobility=50e-4, relative_permittivity=11.68)
p_material_Si = Si_SC(T=T, Na=si(1e16, "cm-3"), electron_diffusion_length=si("290um"),
                hole_mobility=400e-4, relative_permittivity=11.68)

from solcore.solar_cell_solver import default_options as defaults_solcore

options_sc = defaults_solcore
options_sc.optics_method = "external"
options_sc.position = np.arange(0, rtstr.width, options.depth_spacing)
options_sc.light_iv = True
options_sc.wavelength = wl
options_sc.theta = options.theta_in*180/np.pi
V = np.linspace(0, 2, 200)
options_sc.voltages = V

_, diff_absorb_fn = make_absorption_function(profile_rt, rtstr, options, matrix_method=False)

solar_cell = SolarCell(
    [
        Junction([Layer(width=n_material_GaAs_width, material=n_material_GaAs, role='emitter'),
                  Layer(width=p_material_GaAs_width, material=p_material_GaAs, role='base')],
                 sn=2, sp=2, kind='DA'),
        Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                  Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                 sn=1, sp=1, kind='DA')
    ],
    external_reflected=total_R_rt,
    external_absorbed=diff_absorb_fn)

solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(wl*1e9, solar_cell.absorbed, 'k--', label='Absorbed (integrated)')
plt.plot(wl*1e9, solar_cell[0].eqe(wl), 'r-', label='GaAs EQE')
plt.plot(wl*1e9, solar_cell[1].eqe(wl), 'b-', label='Si EQE')
plt.plot(wl*1e9, total_A_rt[:,0], 'r--', label='GaAs - RT')
plt.plot(wl*1e9, total_A_rt[:,1], 'b--', label='Si - RT')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()

plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], '--k', label='Total')
plt.plot(V, -solar_cell[0].iv(V), 'r', label='GaAs')
plt.plot(V, -solar_cell[1].iv(V), 'b', label='Si')
plt.ylim(-20, 250)
plt.xlim(0, 1.8)
plt.legend()
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)') #The expected values of Isc and Voc are 372 A/m^2 and 0.63 V respectively
plt.show()

_, diff_absorb_fn = make_absorption_function(profile_tmm, tmmstr, options)

solar_cell = SolarCell(
    [
        Junction([Layer(width=n_material_GaAs_width, material=n_material_GaAs, role='emitter'),
                  Layer(width=p_material_GaAs_width, material=p_material_GaAs, role='base')],
                 sn=2, sp=2, kind='DA'),
        Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                  Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                 sn=1, sp=1, kind='DA')
    ],
    external_reflected=total_R_tmm,
    external_absorbed=diff_absorb_fn)

solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(wl*1e9, solar_cell.absorbed, 'k--', label='Absorbed (integrated)')
plt.plot(wl*1e9, solar_cell[0].eqe(wl), 'r-', label='GaAs EQE')
plt.plot(wl*1e9, solar_cell[1].eqe(wl), 'b-', label='Si EQE')
plt.plot(wl*1e9, total_A_tmm[:,0], 'r--', label='GaAs - RT')
plt.plot(wl*1e9, total_A_tmm[:,1], 'b--', label='Si - RT')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()


plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], '--k', label='Total')
plt.plot(V, -solar_cell[0].iv(V), 'r', label='GaAs')
plt.plot(V, -solar_cell[1].iv(V), 'b', label='Si')
plt.ylim(-20, 250)
plt.xlim(0, 1.8)
plt.legend()
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)') #The expected values of Isc and Voc are 372 A/m^2 and 0.63 V respectively
plt.show()


_, diff_absorb_fn = make_absorption_function(profile_rcwa, rcwastr, options, False)

solar_cell = SolarCell(
    [
        Junction([Layer(width=n_material_GaAs_width, material=n_material_GaAs, role='emitter'),
                  Layer(width=p_material_GaAs_width, material=p_material_GaAs, role='base')],
                 sn=2, sp=2, kind='DA'),
        Junction([Layer(width=n_material_Si_width, material=n_material_Si, role='emitter'),
                  Layer(width=p_material_Si_width, material=p_material_Si, role='base')],
                 sn=1, sp=1, kind='DA')
    ],
    external_reflected=total_R_rcwa,
    external_absorbed=diff_absorb_fn)

solar_cell_solver(solar_cell, 'qe', options_sc)
solar_cell_solver(solar_cell, 'iv', options_sc)

plt.figure()
plt.plot(wl*1e9, solar_cell.absorbed, 'k--', label='Absorbed (integrated)')
plt.plot(wl*1e9, solar_cell[0].eqe(wl), 'r-', label='GaAs EQE')
plt.plot(wl*1e9, solar_cell[1].eqe(wl), 'b-', label='Si EQE')
plt.plot(wl*1e9, total_A_rcwa[:,0], 'r--', label='GaAs - RT')
plt.plot(wl*1e9, total_A_rcwa[:,1], 'b--', label='Si - RT')
plt.ylim(0,1)
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R/A')
plt.show()


plt.figure()
plt.plot(V, solar_cell.iv['IV'][1], '--k', label='Total')
plt.plot(V, -solar_cell[0].iv(V), 'r', label='GaAs')
plt.plot(V, -solar_cell[1].iv(V), 'b', label='Si')
plt.ylim(-20, 250)
plt.xlim(0, 1.8)
plt.legend()
plt.ylabel('Current (A/m$^2$)')
plt.xlabel('Voltage (V)') #The expected values of Isc and Voc are 372 A/m^2 and 0.63 V respectively
plt.show()