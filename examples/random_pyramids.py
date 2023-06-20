import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import heights_texture
from solcore.constants import q, h, c
from solcore.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from rayflare.textures import regular_pyramids, planar_surface
from solcore import material, si
from solcore.light_source import LightSource
from rayflare.options import default_options
from solcore.structure import Layer
from rayflare.ray_tracing import rt_structure
import seaborn as sns
from cycler import cycler
from solcore.constants import q
from time import time

scan_width = 5.02 # in um

Lx = 1

# use amplitude images, but scale to ZSensor height?
# Not sure whether to use
AFM_pyramids_Z = np.loadtxt("data/Si_PyramidSide_30jan_0001_ZSensor.txt", skiprows=3)

scan_area = AFM_pyramids_Z

noise_angles = np.linspace(0.15, 0.6, 8)
plt.figure()
n_repeats = 1
cols = sns.color_palette("husl", len(noise_angles))

[front, back] = heights_texture(AFM_pyramids_Z*1e6, scan_width, scan_width, coverage_height=0)

P0 = front.P_0s
P1 = front.P_1s
P2 = front.P_2s
N = np.cross(P1 - P0, P2 - P0)

o_t = np.abs(np.real(np.arccos(N[:, 2] / (np.linalg.norm(N, axis=1)))))
plt.hist(o_t, bins=40, range=[0, 1.2], density=True,
         histtype='step',
         color='k',
         label="AFM scan", linestyle='--')
bins_ref = np.histogram(o_t, bins=40, range=[0, 1.2], density=True)

o_t = np.abs(np.real(np.arccos(N[:, 2] / (np.linalg.norm(N, axis=1)))))

plt.hist(o_t, bins=40, range=[0, 1.2], density=True,
         histtype='step',
         color='k',
         label="AFM scan", linestyle='--')
plt.show()

h_dist = np.tan(o_t)*Lx/2

mean_h = np.mean(h_dist)
std_h = np.std(h_dist)

bins_ref = np.histogram(h_dist, bins=15, range=[mean_h-2*std_h, mean_h+2*std_h], density=True)

h_cent = (bins_ref[1][:-1] + bins_ref[1][1:])/2
probs = bins_ref[0]/np.sum(bins_ref[0])


plt.hist(h_dist, bins=15, range=[mean_h-2*std_h, mean_h+2*std_h],
         density=True,
         histtype='step',
         color='k',
         label="AFM scan", linestyle='--')
plt.show()

from rayflare.textures import regular_pyramids

a = regular_pyramids(height_distribution={"p": probs, "h": h_cent})

print(a[0].Points)
a[0].refresh()
print(a[0].Points)


# calibration data

pal = sns.color_palette("husl", 4)

Air = material("Air")()
MgF2 = material("MgF2_Ox")()
IZO = material("IZO_Ox")()
SnO2 = material("SnO2_Ox")()
C60 = material("C60_Ox")()
Perovskite = material("Pvk_Ox_165")()
alumina_Pvk = material("Pvk_9_alumina_1_EMA")()
Me4PACz = material("2PACz")()
ITO = material("ITO_Ox")()
Si = material("Si")()
SiOx = material("SiO")() # don't know stoichiometry, shouldn't matter because really thin, but check
Al2O3 = material("Al2O3_Ox")()
SiNx = material("SiNx_Ox")()
Al = material("Al")()

# build structure

# wavelengths = np.linspace(300, 1200, 40) * 1e-9
wavelengths = np.linspace(250, 1300, 40) * 1e-9


options = default_options()
options.wavelengths = wavelengths
# options.depth_spacing = 1e-9
options.pol = "u"
options.I_thresh = 1e-3
options.randomize_surface = True
options.n_jobs = -4
options.periodic = True
options.project_name = "oxford_Si_doping_nosurf"
options.nx = 40
options.ny = options.nx
options.n_rays = 5*options.nx ** 2
options.lookuptable_angles = 500
options.depth_spacing = 1e-9
options.theta_in = 0*np.pi/180
options.phi_in = 0*np.pi/180

pal = sns.cubehelix_palette(n_colors=10)

cols = cycler('color', pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)
#
# plt.figure()
# for d_SiOx in np.linspace(0, 15, 10):
#     front_materials = [
#         Layer(d_SiOx*1e-9, SiOx),
#     ]
#
#     tmm_str = tmm_structure(front_materials, incidence=Air, transmission=Si)
#
#     res_tmm = tmm_str.calculate(options)
#
#
#     plt.plot(wavelengths*1e9, res_tmm['R'], label=np.round(d_SiOx, 1))
# plt.show()

#
# tmm_struct = tmm_structure(front_materials, incidence=Air, transmission=Si)
#
# res_tmm = tmm_struct.calculate(options)
doping_pyr = 19.2
doping_alk = 20.8
Si_doped_pyr = material("Si_FZ_Green_nk_1e" + str(doping_pyr))()
Si_doped_alk = material("Si_FZ_Green_nk_1e" + str(doping_alk))()

linetype = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

# p-type side is the pyramid side. n-type side is the hemisphere side
d_doped_pyr = 4072.5
d_doped_alk = 1588.8

d_bulk = 150 - (d_doped_pyr + d_doped_alk)/1e3

SiNx.n = interp1d(wavelengths, 0.8*SiNx.n(wavelengths))

pyr_materials = [
    Layer(60e-9, SiNx),
    Layer(20e-9, Al2O3),
    Layer(d_doped_pyr*1e-9, Si_doped_pyr),
]

alk_materials = [
    Layer(d_doped_alk*1e-9, Si_doped_alk),
    Layer(80e-9, SiNx),
]

coherent_back = False

light_source = LightSource(source_type='standard', version='AM1.5g',
                           x=wavelengths,
                           output_units='photon_flux_per_m')

photon_flux = light_source.spectrum(wavelengths)[1]

pal = sns.cubehelix_palette(n_colors=5)

cols = cycler('color', pal)

params = {
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)

front_pyramids_regular = regular_pyramids(
    upright=True,
    elevation_angle=np.mean(o_t)*180/np.pi,
    # height_distribution={"p": probs, "h": h_cent},
    interface_layers=pyr_materials,
    )

front_pyramids_random = regular_pyramids(
    upright=True,
    elevation_angle=np.mean(o_t)*180/np.pi,
    height_distribution={"p": probs, "h": h_cent},
    interface_layers=pyr_materials,
    )

# front_pyramids = regular_pyramids(elevation_angle=49, upright=True, size=1,
#                                        # noise_fraction=0.0267,
#                                        # n_points=20**2,
#                                        # regular_grid=False,
#                                      interface_layers=pyr_materials,
#                                        )

back_alkaline = planar_surface(size=1, interface_layers=alk_materials)

rtstr_random = rt_structure(textures=[front_pyramids_random, back_alkaline],
                        materials= [Si],
                          widths=[d_bulk*1e-6],
                        incidence=Air, transmission=Air,
                        use_TMM=True, options=options,
                              overwrite=True,
                              )
rtstr_regular = rt_structure(textures=[front_pyramids_regular, back_alkaline],
                        materials= [Si],
                          widths=[d_bulk*1e-6],
                        incidence=Air, transmission=Air,
                        use_TMM=True, options=options,
                              overwrite=True,
                              )

start = time()
result_random = rtstr_random.calculate(options)
print("TIME:", time() - start)
start = time()
result_regular = rtstr_regular.calculate(options)
print("TIME", time() - start)

plt.plot(wavelengths * 1e9, result_random['R'], '-k')
plt.plot(wavelengths * 1e9, result_random['T'], '-r')


total_A_pyr = result_random["A_per_layer"][:, 0] + \
              np.sum(result_random["A_per_interface"][0], 1) + \
              np.sum(result_random["A_per_interface"][1], 1)

plt.plot(wavelengths*1e9, total_A_pyr, '-b')

plt.plot(wavelengths*1e9, result_random["A_per_layer"][:, 0], '-b')


plt.plot(wavelengths * 1e9, result_regular['R'], '--k')
plt.plot(wavelengths * 1e9, result_regular['T'], '--r')


total_A_pyr = result_regular["A_per_layer"][:, 0] + \
              np.sum(result_regular["A_per_interface"][0], 1) + \
              np.sum(result_regular["A_per_interface"][1], 1)

plt.plot(wavelengths*1e9, total_A_pyr, '--b')

plt.plot(wavelengths*1e9, result_regular["A_per_layer"][:, 0], '--b')


plt.legend(loc='upper right') #bbox_to_anchor=(1.05, 1))
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbed/reflected fraction')
plt.ylim(0, 1)
# plt.xlim(300, 1300)
# plt.xlim(300, 1200)
# plt.title("Pvk thickness = " + str(np.round(Pvk_thickness)) + " nm")

plt.tight_layout()
plt.show()

plt.figure()
plt.hist(result_random["thetas"][-3], histtype='step',
         color='k', bins=15, density=True, range=[0, np.pi])
plt.hist(result_regular["thetas"][-3], histtype='step',
         color='k', linestyle='--', bins=15, density=True, range=[0, np.pi])
plt.show()

plt.figure()
plt.hist(result_random["thetas"][1], histtype='step',
         color='k', bins=15, density=True, range=[0, np.pi])
plt.hist(result_regular["thetas"][1], histtype='step',
         color='k', linestyle='--', bins=15, density=True, range=[0, np.pi])
plt.show()