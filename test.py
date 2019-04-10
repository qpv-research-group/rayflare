from multiprocessing import freeze_support

from solcore import material
from solcore import si

from structure import Structure, Layer, Interface, Group
from ray_tracing.rt import RTSurface, RT
import numpy as np
import math

import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool

Si = material('Si')()
Air = material('Air')()

char_angle = math.radians(55)
Lx = 1
Ly = 1
h = Lx*math.tan(char_angle)/2
x = np.array([0, Lx/2, Lx, 0, Lx])
y = np.array([0, Ly/2, 0, Ly, Ly])
z = np.array([0, -h, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf = RTSurface(Points)

x = np.array([0, 0, Lx, Lx])
y = np.array([0, Ly, Ly, 0])
z = np.array([0, 0, 0, 0])

Points = np.vstack([x, y, z]).T
surf_back = RTSurface(Points)
wavelengths = si(np.linspace(900, 1160, 20), 'nm')
#pool = Pool(processes = 4)
options =  {'wavelengths': wavelengths, 'I_thresh': 1e-4, 'theta': 0, 'phi': 0,
            'nx': 40, 'ny': 40, 'max_passes': 100, 'parallel': False}#,
            #'pool': pool}

theta = 0*np.pi/180
phi = 0*np.pi/180

a = [Interface(texture = surf), Layer(material = Si, width = si('200um')), \
     Interface(texture = surf_back)]

struct = Group(a, 'RT', depth_spacing = si('1nm'))

z_pos = np.arange(0, 200, 1e-3)

incidence = Air
transmission = Air

group = struct

start = time()
R, T, A_per_layer, profiles, thetas, phis = RT(struct, Air, Air, options)
print(time() - start)
#b = [Interface(texture = surf), Layer(material = Si, width = si('100um')),Interface(texture = surf_back)]
#for i, element in enumerate(a):
#    print(type(element) == Interface, '\n')

A = np.trapz(profiles, z_pos)

A_ref = np.genfromtxt('examples/PVlighthouse_reginv.csv', delimiter = ',')

plt.figure()
plt.plot(wavelengths*1e9, R)
plt.plot(wavelengths*1e9, sum(A_per_layer.T))
plt.plot(A_ref[:,0], A_ref[:,1])
plt.plot(wavelengths*1e9, T)
plt.plot(wavelengths*1e9, R+A+T)
plt.ylim([0,1.02])
#plt.xlim([900, 1160])
plt.legend(['R', 'A', 'A (PV Lighthouse)', 'T'])
plt.show()

plt.figure()
plt.plot(z_pos, profiles[14])

plt.show()


n_theta_bins = 20
n_phi_bins = 20
phi_unique = np.radians(45)
#data = data[~(data == None)]
bins = np.linspace(-1e-9, np.pi+1e-9, n_theta_bins+1)
digitized_theta = np.digitize(thetas, bins, right = True)

phi_bins = np.linspace(-1e-9, (phi_unique+1e-9), n_phi_bins+1)

# need to go from -pi -> pi to 0 -> 2pi;

phi_fold = (phis + 2*np.pi*(phis//np.pi))%phi_unique

digitized_phi = np.digitize(phi_fold, phi_bins, right = True)

