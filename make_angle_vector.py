import numpy as np
import xarray as xr

options = {'n_angle_bins': 100, 'c_azimuth': 0.25, 'phi_symmetry': np.pi/4}

n_angle_bins = options['n_angle_bins']
c_azimuth = options['c_azimuth']
phi_sym = options['phi_symmetry']

sin_a_b = np.linspace(0, 1, n_angle_bins+1) # number of bins is between 0 and 90 degrees
# even spacing in terms of sin(theta) rather than theta
# will have the same number of bins between 90 and 180 degrees

theta_intv= np.concatenate([np.arcsin(sin_a_b), np.pi-np.flip(np.arcsin(sin_a_b[:-1]))])

theta_middle = (theta_intv[:-1] + theta_intv[1:])/2
phi_intv = []
angle_vector = np.empty((0,3))

for i1, theta in enumerate(theta_middle):
    if theta > np.pi/2:
        ind = len(theta_intv)-(i1 + 1) # + 1 because Python is zero-indexed
    else:
        ind = i1 + 1

    phi_intv.append(np.linspace(0, phi_sym, np.ceil(c_azimuth*ind)+1))
    phi_middle = (phi_intv[i1][:-1] + phi_intv[i1][1:])/2

    angle_vector = np.append(angle_vector, np.array([np.array(len(phi_middle)*[i1]),
                                  np.array(len(phi_middle)*[theta]),
                                      phi_middle]).T, axis = 0)


# folding phi back into first quadrant: need to make sure it is positive, so 0 to 360
# not -180 to + 180. Then just do phi % phi_sym

# -180 to +180 -> 0 to 360 degrees:
# abs(angle//180)*360 + angle
# full transformation: (abs(angle//np.pi)*2*np.pi + angle) % phi_sym