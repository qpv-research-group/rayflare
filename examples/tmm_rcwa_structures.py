import numpy as np
from solcore import si, material
from solcore.structure import Layer
from solcore.constants import q, h, c
from solcore.interpolate import interp1d
from solcore.solar_cell import SolarCell

from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from rayflare.options import default_options

import matplotlib.pyplot as plt

InGaP = material('GaInP')(In=0.5)
GaAs = material('GaAs')()
Ge = material('Ge')()
Ag = material('Ag')()
Air = material('Air')()

Al2O3 = material('Al2O3')()

wavelengths = np.linspace(250, 1900, 500) * 1e-9

options = default_options()

options.wavelengths = wavelengths
options.orders = 2

size = ((100, 0), (0, 100))

# anti-reflection coating
ARC = [Layer(si('80nm'), Al2O3)]

solar_cell = SolarCell(ARC + [Layer(material=InGaP, width=si('400nm')),
                               Layer(material=GaAs, width=si('4000nm')),
                               Layer(material=Ge, width=si('3000nm'))], substrate=Ag)


rcwa_setup = rcwa_structure(solar_cell, size=size, options=options, incidence=Air, transmission=Ag)
tmm_setup = tmm_structure(solar_cell, incidence=Air, transmission=Ag, no_back_reflection=False)

spect = np.loadtxt('data/AM0.csv', delimiter=',')

AM0 = interp1d(spect[:, 0], spect[:, 1])(wavelengths * 1e9)

for pol in ['s', 'p', 'u']:
    for angle in [0, np.pi/3]:

        options['pol'] = pol
        options['theta_in'] = angle

        rcwa_result = rcwa_setup.calculate(options)
        tmm_result = tmm_setup.calculate(options)

        Jsc_TMM = 0.1 * (q / (h * c))* np.trapz(wavelengths[:, None]*1e9*tmm_result['A_per_layer'] * AM0[:, None],
                                                    wavelengths*1e9, axis=0)/1e9

        Jsc_RCWA = 0.1 * (q / (h * c))* np.trapz(wavelengths[:, None]*1e9*rcwa_result['A_per_layer'] * AM0[:, None],
                                                    wavelengths*1e9, axis=0)/1e9

        print('Pol: ' + options['pol'] + ', Angle: ' + str(options['theta_in']) + ' deg \n' +
        'TMM currents: ' + str(np.round(Jsc_TMM[1:], 3)) + '\n' +
        'RCWA currents: ' + str(np.round(Jsc_RCWA[1:], 3)) + '\n')


        plt.figure()

        plt.plot(wavelengths*1e9, tmm_result['A_per_layer'][:, 1:])
        plt.plot(wavelengths*1e9, rcwa_result['A_per_layer'][:, 1:], '--')
        plt.title('Pol: ' + options['pol'] + ', Angle: ' + str(options['theta_in']) + ' deg')
        plt.plot(wavelengths*1e9, tmm_result['R'])
        plt.plot(wavelengths*1e9, rcwa_result['R'], '--')
        plt.plot(wavelengths*1e9, tmm_result['T'])
        plt.plot(wavelengths*1e9, rcwa_result['T'], '--')
        plt.xlabel('Wavelength nm')
        plt.ylabel('Absorption per layer/Reflection')
        plt.show()