from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.rigorous_coupled_wave_analysis.rcwa import initialise_S
from rayflare.options import default_options
from solcore import material
from solcore.structure import Layer
import numpy as np
import matplotlib.pyplot as plt

Si = material('Si')()
Air = material('Air')()
size = ((700, 0), (0, 700)) # 700 nm square unit cell (doesn't matter for planar layers)

layers = []

options = default_options()
options.wavelength = np.linspace(300, 1000, 60) * 1e-9
orders = 2

rcwa_strt = rcwa_structure(layers, size, options, Air, Si)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

for angle in [45, 55, 65]:
    r_s = np.zeros_like(options.wavelength, dtype=complex)
    r_p = np.zeros_like(options.wavelength, dtype=complex)

    for wl_ind, wl in enumerate(options.wavelength):

        S = initialise_S(rcwa_strt.size, orders, rcwa_strt.geom_list, rcwa_strt.layers_oc[wl_ind],
                            rcwa_strt.shapes_oc[wl_ind], rcwa_strt.shapes_names, rcwa_strt.widths,
                         options.S4_options)
        actual_orders = len(S.GetBasisSet()) # get the actual number of orders S4 chose to use
        S.SetExcitationPlanewave((angle, 0), 1, 0, 0)
        S.SetFrequency(1 / wl)

        (forw, back) = S.GetAmplitudes('layer_1')

        # this isn't clear in the document, but I think this is indexed as: for N orders,
        # the first N elements in "forw" are the forward-travelling E-field amplitudes in each
        # order, while the elements N:end are the H-field amplitudes (similarly for "back", but
        # for the backwards-travelling E and H field amplitudes).

        r_s[wl_ind] = back[0]/forw[0]

        S = initialise_S(rcwa_strt.size, actual_orders, rcwa_strt.geom_list, rcwa_strt.layers_oc[wl_ind],
                            rcwa_strt.shapes_oc[wl_ind], rcwa_strt.shapes_names, rcwa_strt.widths,
                         options.S4_options)
        S.SetExcitationPlanewave((angle, 0), 0, 1, 0)
        S.SetFrequency(1 / wl)
        (forw, back) = S.GetAmplitudes('layer_1')

        # E field amplitudes are zero for p-polarised light, need to use H field amplitudes
        H_forw = forw[actual_orders:]
        H_back = back[actual_orders:]

        r_p[wl_ind] = H_back[0]/H_forw[0]

    # Fresnel equations:

    n1 = np.sqrt(rcwa_strt.layers_oc[:,0])
    n2 = np.sqrt(rcwa_strt.layers_oc[:,1])

    theta_t = np.arcsin(n1*np.sin(angle*np.pi/180)/n2)
    
    r_s_F = (n1*np.cos(angle*np.pi/180) - n2*np.cos(theta_t)) / (n1*np.cos(angle*np.pi/180) + n2*np.cos(theta_t))
    r_p_F = (n2*np.cos(angle*np.pi/180) - n1*np.cos(theta_t)) / (n2*np.cos(angle*np.pi/180) + n1*np.cos(theta_t))

    rho = r_p/r_s

    rho_F = r_p_F/r_s_F

    rho_mag = np.abs(rho)
    delta = np.angle(rho)
    psi = np.arctan(rho_mag)

    rho_mag_F = np.abs(rho_F)
    delta_F = np.angle(rho_F)
    psi_F = np.arctan(rho_mag_F)

    ax1.plot(options.wavelength*1e9, 180*rho_mag/np.pi, label=str(angle) + ' (RCWA)')
    ax2.plot(options.wavelength*1e9, 180 - delta*180/np.pi)
    # not completely sure why I need to do 180 - angle (or take the negative below for rho)

    ax1.plot(options.wavelength*1e9, 180*rho_mag_F/np.pi, '--', label=str(angle) + ' (Fresnel)')
    ax2.plot(options.wavelength*1e9, -delta_F*180/np.pi, '--')

ax1.legend()
ax1.set_ylabel('Psi (degrees)')
ax2.set_ylabel('Delta (degrees)')
ax2.set_xlabel('Wavelength (nm)')
ax1.set_title('Planar Air/Si')
plt.show()

r_s_65 = r_s
r_p_65 = r_p

# try it with some kind of grating

Si = material('Si')()
Air = material('Air')()

layers = [Layer(1e-6, Si,
                geometry=[{"type": "rectangle", "mat": Air,
                           "center": (500, 500), "halfwidths": (100, 100), "angle": 45}])]

orders = 60

rcwa_strt = rcwa_structure(layers, size, options, Air, Si)

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

for angle in [45, 55, 65]:
    r_s = np.zeros_like(options.wavelength, dtype=complex)
    r_p = np.zeros_like(options.wavelength, dtype=complex)

    for wl_ind, wl in enumerate(options.wavelength):
        S = initialise_S(rcwa_strt.size, orders, rcwa_strt.geom_list, rcwa_strt.layers_oc[wl_ind],
                         rcwa_strt.shapes_oc[wl_ind], rcwa_strt.shapes_names, rcwa_strt.widths,
                         options.S4_options)
        actual_orders = len(S.GetBasisSet())
        S.SetExcitationPlanewave((angle, 0), 1, 0, 0)
        S.SetFrequency(1 / wl)

        (forw, back) = S.GetAmplitudes('layer_1')

        # this isn't clear in the document, but I think this is indexed as: for N orders,
        # the first N elements in "forw" are the forward-travelling E-field amplitudes in each
        # order, while the elements N:end are the H-field amplitudes (similarly for "back", but
        # for the backwards-travelling E and H field amplitudes).

        r_s[wl_ind] = back[0] / forw[0]

        S = initialise_S(rcwa_strt.size, actual_orders, rcwa_strt.geom_list, rcwa_strt.layers_oc[wl_ind],
                         rcwa_strt.shapes_oc[wl_ind], rcwa_strt.shapes_names, rcwa_strt.widths,
                         options.S4_options)
        S.SetExcitationPlanewave((angle, 0), 0, 1, 0)
        S.SetFrequency(1 / wl)
        (forw, back) = S.GetAmplitudes('layer_1')

        # E field amplitudes are zero for p-polarised light, need to use H field amplitudes
        H_forw = forw[actual_orders:]
        H_back = back[actual_orders:]

        r_p[wl_ind] = H_back[0] / H_forw[0]

    # Fresnel equations:

    n1 = np.sqrt(rcwa_strt.layers_oc[:, 0])
    n2 = np.sqrt(rcwa_strt.layers_oc[:, 2])

    theta_t = np.arcsin(n1 * np.sin(angle * np.pi / 180) / n2)

    r_s_F = (n1 * np.cos(angle * np.pi / 180) - n2 * np.cos(theta_t)) / (
                n1 * np.cos(angle * np.pi / 180) + n2 * np.cos(theta_t))
    r_p_F = (n2 * np.cos(angle * np.pi / 180) - n1 * np.cos(theta_t)) / (
                n2 * np.cos(angle * np.pi / 180) + n1 * np.cos(theta_t))

    rho = r_p / r_s

    rho_F = r_p_F / r_s_F

    rho_mag = np.abs(rho)
    delta = np.angle(rho)
    psi = np.arctan(rho_mag)

    rho_mag_F = np.abs(rho_F)
    delta_F = np.angle(rho_F)
    psi_F = np.arctan(rho_mag_F)

    ax1.plot(options.wavelength * 1e9, 180 * rho_mag / np.pi, label=str(angle) + ' (RCWA)')
    ax2.plot(options.wavelength * 1e9, 180 - delta * 180 / np.pi)

    ax1.plot(options.wavelength * 1e9, 180 * rho_mag_F / np.pi, '--',
             label=str(angle) + ' (Fresnel)')
    ax2.plot(options.wavelength * 1e9, -delta_F * 180 / np.pi, '--')

ax1.legend()
ax1.set_ylabel('Psi (degrees)')
ax2.set_ylabel('Delta (degrees)')
ax2.set_xlabel('Wavelength (nm)')
ax1.set_title('Air/Si with grating')
plt.show()

r_s_65_grating = r_s
r_p_65_grating = r_p

R_zeroorder_nograting = 0.5*(np.abs(r_s_65)**2 + np.abs(r_p_65)**2)
R_zeroorder_grating = 0.5*(np.abs(r_s_65_grating)**2 + np.abs(r_p_65_grating)**2)

plt.figure()
plt.plot(options.wavelength*1e9, R_zeroorder_nograting, label='Planar')
plt.plot(options.wavelength*1e9, R_zeroorder_grating, '--', label='Grating')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('R (zeroth order)')
plt.show()

