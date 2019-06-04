import numpy as np
import numpy.matlib
import os
from scipy.spatial import Delaunay
from cmath import sin, cos, asin, sqrt, acos, atan
from math import atan2, ceil
from random import random
import matplotlib.pyplot as plt
from itertools import product
from functools import partial
import xarray as xr
from angles import fold_phi, make_angle_vector
from sparse import COO, save_npz, stack
from config import results_path
from joblib import Parallel, delayed

# want a Delaunay-type triangulation object which has the same attributes as
# the return from scipy.spatial.Delaunaytheta_t = np.asin((n1/n2)*np.sin(theta)), but contains 3D rather than 2D coordinates,
# i.e. a 2D surface in 3D space.

def RT(group, incidence, transmission, options):

    surf_name = options['surf_name']
    front_or_rear = options['front_or_rear']
    prof_indices = options['prof_indices']
    structpath = os.path.join(results_path, options['struct_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)

    wavelengths = options['wavelengths']
    n_rays = options['n_rays']
    nx = options['nx']
    ny = options['ny']
    n_angles = int(np.ceil(n_rays/(nx*ny)))

    phi_sym = options['phi_symmetry']
    n_theta_bins = options['n_theta_bins']
    c_az = options['c_azimuth']
    pol = options['pol']
    Fr_or_TMM = options['Fr_or_TMM']
    n_absorbing_layers = options['n_absorbing_layers']


    if Fr_or_TMM == 1:
        lookuptable = xr.open_dataset(os.path.join(structpath, surf_name + '.nc'))
        if front_or_rear == 'rear':
            lookuptable = lookuptable.assign_coords(side=np.flip(lookuptable.side))
    else:
        lookuptable = None

    theta_intv, phi_intv, angle_vector = make_angle_vector(n_theta_bins, phi_sym, c_az)

    if options['random_angles']:
        thetas_in = np.random.random(n_angles)*np.pi/2
        phis_in = np.random.random(n_angles)*2*np.pi
    else:
        angles_in = angle_vector[:int(len(angle_vector)/2),:]
        n_reps = int(np.ceil(n_angles/len(angles_in)))
        thetas_in = np.tile(angles_in[:,1], n_reps)[:n_angles]
        phis_in = np.tile(angles_in[:,2], n_reps)[:n_angles]

    mats = [incidence]
    for i1 in range(len(group.materials)):
        mats.append(group.materials[i1])
    mats.append(transmission)

    surfaces = group.textures

    I_thresh = options['I_thresh']

    widths = group.widths[:]
    widths.insert(0, 0)
    widths.append(0)
    widths = 1e6*np.array(widths)  # convert to um

    z_space = 1e6*group.depth_spacing
    z_pos = np.arange(0, sum(widths), z_space)

    nks = np.empty((len(mats), len(wavelengths)), dtype=complex)
    alphas = np.empty((len(mats), len(wavelengths)), dtype=complex)

    for i1, mat in enumerate(mats):
        nks[i1] = mat.n(wavelengths) + 1j*mat.k(wavelengths)
        alphas[i1] = mat.k(wavelengths)*4*np.pi/(wavelengths*1e6)

    h = max(surfaces[0].Points[:, 2])
    x_lim = surfaces[0].Lx
    y_lim = surfaces[0].Ly

    xs = np.linspace(x_lim/100, x_lim-(x_lim/100), nx)
    ys = np.linspace(y_lim/100, y_lim-(y_lim/100), ny)

    #allArrays = [RT_wl(i1, n_angles, nx, ny, z_pos, widths, thetas_in, phis_in, h, xs, ys, nks, alphas, surfaces,
    #      I_thresh, pol, phi_sym, theta_intv, phi_intv, angle_vector) for i1 in range(len(wavelengths))]
    if options['parallel']:
        allres = Parallel(n_jobs=options['n_jobs'])(delayed(RT_wl)
                                       (i1, wavelengths[i1], n_angles, nx, ny, z_pos,
                                        widths, thetas_in, phis_in, h,
                                        xs, ys, nks, alphas, surfaces,
                                        I_thresh, pol, phi_sym, theta_intv,
                                        phi_intv, angle_vector, Fr_or_TMM, n_absorbing_layers,
                                        lookuptable, prof_indices)
                                   for i1 in range(len(wavelengths)))

    else:
        allres = [RT_wl(i1, wavelengths[i1], n_angles, nx, ny, z_pos, widths,
                        thetas_in, phis_in, h, xs, ys, nks, alphas, surfaces,
                                 I_thresh, pol, phi_sym, theta_intv, phi_intv,
                        angle_vector, Fr_or_TMM, n_absorbing_layers, lookuptable, prof_indices)
                  for i1 in range(len(wavelengths))]

    allArrays = stack([item[0] for item in allres])
    print(allArrays.shape)
    absArrays = stack([item[1] for item in allres])

    structpath = os.path.join(results_path, options['struct_name'])
    if not os.path.isdir(structpath):
        os.mkdir(structpath)
    savepath_RT = os.path.join(structpath, surf_name + front_or_rear + 'RT.npz')
    savepath_A = os.path.join(structpath, surf_name + front_or_rear + 'A.npz')
    save_npz(savepath_RT, allArrays)
    save_npz(savepath_A, absArrays)

    if len(prof_indices) > 0:
        local_angles = stack([item[2] for item in allres])
        savepath_prof = os.path.join(structpath, surf_name + front_or_rear + 'Aprof.npz')
        save_npz(savepath_prof, local_angles)
        return allArrays, absArrays, local_angles
    else:
        return allArrays, absArrays


def RT_wl(i1, wl, n_angles, nx, ny, z_pos, widths, thetas_in, phis_in, h, xs, ys, nks, alphas, surfaces,
          I_thresh, pol, phi_sym, theta_intv, phi_intv, angle_vector, Fr_or_TMM, n_abs_layers, lookuptable, prof_indices):
    print('wavelength = ', wl)
    #n_prof_layers = len(prof_indices)
    theta_out = np.zeros((n_angles, nx * ny))
    phi_out = np.zeros((n_angles, nx * ny))
    #absorption_profiles = np.zeros((n_angles, len(z_pos)))
    A_surface_layers = np.zeros((n_angles, nx*ny, n_abs_layers))
    theta_local_incidence = np.zeros((n_angles, nx*ny))
    # R = np.zeros(len(wavelengths))
    # T = np.zeros(len(wavelengths))
    #A_layer = np.zeros((n_angles, len(widths)))
    for i2 in range(n_angles):
        # print(i2)
        theta = thetas_in[i2]
        phi = phis_in[i2]
        r = abs((h + 1) / cos(theta))
        r_a_0 = np.real(np.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]))
        for c, vals in enumerate(product(xs, ys)):
            I, th_o, phi_o, surface_A = \
                single_ray_interface(vals[0], vals[1], nks[:, i1], alphas[:, i1],
                           r_a_0, theta, phi, surfaces, widths, z_pos,
                           I_thresh, pol, wl, Fr_or_TMM, lookuptable)
            #absorption_profiles[i2] = absorption_profiles[i2] + profile / (nx * ny)
            theta_out[i2, c] = th_o
            phi_out[i2, c] = phi_o
            #A_layer[i2] = A_layer[i2] + A_per_layer / (nx * ny)
            A_surface_layers[i2, c] = surface_A[0]
            theta_local_incidence[i2, c] = np.real(surface_A[1])

    print('done ray-tracing, binning')
    phi_out = fold_phi(phi_out, phi_sym)
    phis_in = fold_phi(phis_in, phi_sym)

    theta_out = abs(theta_out)
    theta_local_incidence = np.abs(theta_local_incidence)
    n_thetas = len(theta_intv) - 1

    if n_abs_layers > 0:
        # now we need to make bins for the absorption
        theta_intv = np.append(theta_intv, 11)
        phi_intv = phi_intv + [np.array([0])]

    # xarray: can use coordinates in calculations using apply!
    binned_theta_in = np.digitize(thetas_in, theta_intv, right=True) - 1

    binned_theta_out = np.digitize(theta_out, theta_intv, right=True) - 1
    # -1 to give the correct index for the bins in phi_intv

    phi_in = xr.DataArray(phis_in,
                          coords={'theta_bin': (['angle_in'], binned_theta_in)},
                          dims=['angle_in'])

    bin_in = phi_in.groupby('theta_bin').apply(overall_bin,
                                               args=(phi_intv, angle_vector[:, 0])).data

    phi_out = xr.DataArray(phi_out,
                           coords={'theta_bin': (['angle_in', 'position'], binned_theta_out)},
                           dims=['angle_in', 'position'])

    bin_out = phi_out.groupby('theta_bin').apply(overall_bin,
                                                 args=(phi_intv, angle_vector[:, 0])).data

    out_mat = np.zeros((len(angle_vector), int(len(angle_vector) / 2)))
    # everything is coming in from above so we don't need 90 -> 180 in incoming bins
    A_mat = np.zeros((n_abs_layers, int(len(angle_vector)/2)))

    n_rays_in_bin = np.zeros(int(len(angle_vector) / 2))
    n_rays_in_bin_abs = np.zeros(int(len(angle_vector) / 2))

    binned_local_angles = np.digitize(theta_local_incidence, theta_intv, right=True) - 1
    local_angle_mat = np.zeros((int((len(theta_intv) -1 )/ 2), int(len(angle_vector) / 2)))

    for l1 in range(len(thetas_in)):
        for l2 in range(nx * ny):
            n_rays_in_bin[bin_in[l1]] += 1
            #print(thetas_in[l1], bin_in[l1], bin_out[l1,l2])
            if binned_theta_out[l1, l2] <= (n_thetas-1):
                # reflected or transmitted
                out_mat[bin_out[l1, l2], bin_in[l1]] += 1

            else:
                # absorbed in one of the surface layers
                n_rays_in_bin_abs[bin_in[l1]] += 1
                per_layer = A_surface_layers[l1, l2]
                A_mat[:, bin_in[l1]] += per_layer
                local_angle_mat[binned_local_angles[l1, l2], bin_in[l1]] += 1

    # normalize
    out_mat = out_mat/n_rays_in_bin
    overall_abs_frac = n_rays_in_bin_abs/n_rays_in_bin
    abs_scale = overall_abs_frac/np.sum(A_mat, 0)
    A_mat = abs_scale*A_mat
    out_mat[np.isnan(out_mat)] = 0
    A_mat[np.isnan(A_mat)] = 0

    out_mat = COO(out_mat)  # sparse matrix
    A_mat = COO(A_mat)
    #savepath = os.path.join(structpath, surf_name + '_' + "{0:.6g}".format(wl * 1e9) + '.npz')
    #save_npz(savepath, result)
    if n_abs_layers > 0:
        local_angle_mat = local_angle_mat/np.sum(local_angle_mat, 0)
        local_angle_mat[np.isnan(local_angle_mat)] = 0
        local_angle_mat = COO(local_angle_mat)
        return out_mat, A_mat, local_angle_mat

    else:
        return out_mat, A_mat

def normalize(x):
    if sum(x > 0):
        x = x/sum(x) - x.coords['A']
    return x

def overall_bin(x, phi_intv, angle_vector_0):
    #print(x.coords['theta_bin'])
    phi_ind = np.digitize(x, phi_intv[x.coords['theta_bin'].data[0]], right=True) - 1
    bin = np.argmin(abs(angle_vector_0 - x.coords['theta_bin'].data[0])) + phi_ind
    return bin

class RTSurface:
    def __init__(self, Points):
        # want the 'base' of the scan to lie on the plane z = 0
        # so if min(Points(:, 3) < 0 or > 0, translate

        # Points[:, 2] = Points[:, 2] - min(Points[:,2])
        tri = Delaunay(Points[:, [0, 1]])
        self.simplices = tri.simplices
        self.Points = Points
        self.P_0s = Points[tri.simplices[:, 0]]
        self.P_1s = Points[tri.simplices[:, 1]]
        self.P_2s = Points[tri.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.size = self.P_0s.shape[0]
        self.Lx = abs(min(Points[:, 0])-max(Points[:, 0]))
        self.Ly = abs(min(Points[:, 1])-max(Points[:, 1]))
        self.z_min = min(Points[:, 2])
        self.z_max = max(Points[:, 2])

        self.zcov= Points[:,2][np.all(np.array([Points[:,0] == min(Points[:,0]), Points[:,1] == min(Points[:,1])]), axis = 0)]
        #z1 = Points[:,2][np.all(np.array([Points[:,0] == min(Points[:,0]), Points[:,1] == max(Points[:,1])]), axis = 0)]
        #z2 = Points[:,2][np.all(np.array([Points[:,0] == max(Points[:,0]), Points[:,1] == min(Points[:,1])]), axis = 0)]
        #z3 = Points[:,2][np.all(np.array([Points[:,0] == max(Points[:,0]), Points[:,1] == max(Points[:,1])]), axis = 0)]

def calc_R(n1, n2, theta, pol):
    theta_t = np.arcsin((n1/n2)*np.sin(theta))
    Rs = np.abs((n1*np.cos(theta)-n2*np.cos(theta_t))/(n1*np.cos(theta)+n2*np.cos(theta_t)))**2
    Rp = np.abs((n1 * np.cos(theta_t) - n2 * np.cos(theta)) / (n1 * np.cos(theta_t) + n2 * np.cos(theta)))**2
    if pol == 's':
        return Rs
    if pol == 'p':
        return Rp
    else:
        return (Rs+Rp)/2

def exit_side(r_a, d, Lx, Ly):
    n = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]])    # surface normals: top, right, bottom, left
    p_0 = np.array([[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]])  # points on each plane
    t = np.sum((p_0 - r_a) * n, axis=1) / np.sum(d * n, axis=1)     # r_intersect = r_a + t*d
    which_intersect = t > 0                                         # only want intersections of forward-travelling ray
    t[~which_intersect] = float('inf')                              # set others to inf to avoid finding when doing min
    which_side = np.argmin(t)                                       # find closest plane

    return which_side, t[which_side]


def calc_angle(x):

    v1 = np.array([0, 1])
    return np.math.atan2(np.linalg.det([x, v1]), np.dot(x, v1))  # - 180 to 180

def single_ray_stack(x, y,  nks, alphas, r_a_0, theta, phi, surfaces, widths, z_pos, I_thresh, pol, wl, Fr_or_TMM, lookuptable):
    # final_res = 0: reflection
    # final_res = 1: transmission
    # This should get a list of surfaces and materials (optical constants, alpha + widths); there is one less surface than material
    # minimum is one surface and two materials
    # direction = 1: travelling down
    # direction = -1: travelling up

    #      incidence medium (0)
    # surface 0  ---------
    #           material 1
    # surface 1  ---------
    #           material 2
    # surface 2  ---------

    # should end when either material = final material (len(materials)-1) & direction == 1 or
    # material = 0 & direction == -1
    profile = np.zeros(len(z_pos))
    # do everything in microns
    A_per_layer = np.zeros(len(widths))

    direction = 1   # start travelling downwards; 1 = down, -1 = up
    mat_index = 0   # start in first medium
    surf_index = 0
    stop = False
    I = 1

    # could be done before to avoid recalculating every time
    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array([x, y, 0])           # set r_a and r_b so that ray has correct angle & intersects with first surface
    d = (r_b - r_a) / np.linalg.norm(r_b - r_a) # direction (unit vector) of ray
    #ray_path = np.append(r_a, 0)
    n_passes = 0

    depths = []
    depth_indices = []
    for i1 in range(len(widths)):
        depth_indices.append((z_pos < np.cumsum(widths)[i1]) & (z_pos >= np.cumsum(widths)[i1 - 1]))
        depths.append(z_pos[depth_indices[i1]] - \
             np.cumsum(widths)[i1 - 1])

    #print('nwidths', len(widths))
    #print('s/d before', surf_index, direction, mat_index, stop)

    while not stop:
        #ax.plot([r_a[0], r_a[0] + d[0]], [r_a[1], r_a[1] + d[1]], [r_a[2], r_a[2] + d[2]])
        # translate r_a so that the ray can actually intersect with the unit cell of the surface it is approaching

        surf = surfaces[surf_index]

        r_a[0] = r_a[0]-surf.Lx*((r_a[0]+d[0]*(surf.zcov-r_a[2])/d[2])//surf.Lx)
        r_a[1] = r_a[1]-surf.Ly*((r_a[1]+d[1]*(surf.zcov-r_a[2])/d[2])//surf.Ly)

        #ray_path = np.vstack((ray_path, np.append(r_a + d*(((surf.zcov - r_a[2])/d[2])-2), 1)))
        #print(r_a, d)
        res, theta, phi, r_a, d, theta_loc = single_interface_check(r_a, d, nks[mat_index],
                                     nks[mat_index+1], surf, surf.Lx, surf.Ly, direction,
                                     surf.zcov, pol, wl, Fr_or_TMM, lookuptable)

        if res == 0:  # reflection
            direction = -direction # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

            I_b = I
            # print(mat_index)
            DA, stop, I, theta = traverse(widths[mat_index], theta, alphas[mat_index], x, y, I,
                                          depths[mat_index], I_thresh, direction)
            A_per_layer[mat_index] = A_per_layer[mat_index] + I_b - I
            profile[depth_indices[mat_index]] = profile[depth_indices[mat_index]] + DA

            n_passes = n_passes + 1
            surface_A = [0, 10]

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction # is this right?

            I_b = I
            # print(mat_index)
            DA, stop, I, theta = traverse(widths[mat_index], theta, alphas[mat_index], x, y, I,
                                          depths[mat_index], I_thresh, direction)
            A_per_layer[mat_index] = A_per_layer[mat_index] + I_b - I
            profile[depth_indices[mat_index]] = profile[depth_indices[mat_index]] + DA

            n_passes = n_passes + 1
            surface_A = [0, 10]

        if res == 2:
            surface_A = [theta, theta_loc] # passed a list of absorption per layer in theta
            stop = True
            theta = 10 # theta is actually list of absorption per layer

        if direction == 1 and mat_index == (len(widths)-1):
            #print('transmission')
            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True

        #print('s/d after', surf_index, direction, mat_index, stop)

    #print('n_passes', n_passes)
    return I, profile, A_per_layer, theta, phi, surface_A

def single_ray_interface(x, y,  nks, alphas, r_a_0, theta, phi, surfaces, widths, z_pos, I_thresh, pol, wl, Fr_or_TMM, lookuptable):
    # final_res = 0: reflection
    # final_res = 1: transmission
    # This should get a list of surfaces and materials (optical constants, alpha + widths); there is one less surface than material
    # minimum is one surface and two materials
    # direction = 1: travelling down
    # direction = -1: travelling up

    #      incidence medium (0)
    # surface 0  ---------
    #           material 1
    # surface 1  ---------
    #           material 2
    # surface 2  ---------

    # should end when either material = final material (len(materials)-1) & direction == 1 or
    # material = 0 & direction == -1

    direction = 1   # start travelling downwards; 1 = down, -1 = up
    mat_index = 0   # start in first medium
    surf_index = 0
    stop = False
    I = 1

    # could be done before to avoid recalculating every time
    r_a = r_a_0 + np.array([x, y, 0])
    r_b = np.array([x, y, 0])           # set r_a and r_b so that ray has correct angle & intersects with first surface
    d = (r_b - r_a) / np.linalg.norm(r_b - r_a) # direction (unit vector) of ray
    #ray_path = np.append(r_a, 0)
    n_passes = 0



    while not stop:

        surf = surfaces[surf_index]

        r_a[0] = r_a[0]-surf.Lx*((r_a[0]+d[0]*(surf.zcov-r_a[2])/d[2])//surf.Lx)
        r_a[1] = r_a[1]-surf.Ly*((r_a[1]+d[1]*(surf.zcov-r_a[2])/d[2])//surf.Ly)

        res, theta, phi, r_a, d, theta_loc = single_interface_check(r_a, d, nks[mat_index],
                                     nks[mat_index+1], surf, surf.Lx, surf.Ly, direction,
                                     surf.zcov, pol, wl, Fr_or_TMM, lookuptable)

        if res == 0:  # reflection
            direction = -direction # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

            I_b = I
            # print(mat_index)

            surface_A = [0, 10]

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction # is this right?

            I_b = I
            # print(mat_index)

            surface_A = [0, 10]

        if res == 2:
            surface_A = [theta, theta_loc] # passed a list of absorption per layer in theta
            stop = True
            theta = 10 # theta is actually list of absorption per layer

        if direction == 1 and mat_index == (len(widths)-1):
            #print('transmission')
            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True

        #print('s/d after', surf_index, direction, mat_index, stop)

    #print('n_passes', n_passes)
    return I, theta, phi, surface_A


def traverse(width, theta, alpha, x, y, I_i, positions, I_thresh, direction):
    # Want to get absorption PROFILE
    #print('traversing', width, alpha)
    stop = False
    DA = (alpha/abs(cos(theta)))*I_i*np.exp((-alpha*positions/abs(cos(theta))))

    I_back = I_i*np.exp(-alpha*width/abs(cos(theta)))

    if I_back < I_thresh:
        #print('absorbed')
        stop = True
        theta = None

    if direction == -1:
        #print('flipping DA because going up')
        DA = np.flip(DA)
    #if len(positions) > 0:
        #print('theta, alpha, I_i, DA[200]', theta, alpha, I_i, DA[200], direction)
    return DA, stop, I_back, theta


def decide_RT_Fresnel(n0, n1, theta, d, N, side, pol, rnd, wl = None, lookuptable = None):
    R = calc_R(n0, n1, theta, pol)

    if rnd <= R:  # REFLECTION

        d = np.real(d - 2 * np.dot(d, N) * N)
        d = d / np.linalg.norm(d)

    else:  # TRANSMISSION
        # transmission, refraction
        # for now, ignore effect of k on refraction

        tr_par = ((np.real(n0) / np.real(n1)) ** side) * (d - np.dot(d, N) * N)

        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N

        side = -side
        d = np.real(tr_par + tr_perp)
        d = d / np.linalg.norm(d)

    return d, side, None # never absorbed, A = False

def decide_RT_TMM(n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable):
    #data = lookuptable.loc[dict(side=side, pol=pol)].interp(wl=wl*1e9, angle=abs(theta))
    data = lookuptable.loc[dict(side=side, pol=pol)].sel(angle=abs(theta), wl=wl*1e9, method='nearest')
    R = np.real(data['R'].data.item(0))
    T = np.real(data['T'].data.item(0))
    A_per_layer = np.real(data['Alayer'].data)
    A_params = np.real(data['Aprof'].data)

    #print('side, pol, wl, theta', side, pol, wl, theta)

    if rnd <= R:  # REFLECTION

        d = np.real(d - 2 * np.dot(d, N) * N)
        d = d / np.linalg.norm(d)
        A = None

    elif (rnd > R) & (rnd <= (R+T)):   # TRANSMISSION
        # transmission, refraction
        # for now, ignore effect of k on refraction

        tr_par = ((np.real(n0) / np.real(n1)) ** side) * (d - np.dot(d, N) * N)

        tr_perp = -sqrt(1 - np.linalg.norm(tr_par) ** 2) * N

        side = -side
        d = np.real(tr_par + tr_perp)
        d = d / np.linalg.norm(d)
        A = None

    else:
        # absorption
        A = A_per_layer

    #print('R, T, rand, absorbed?', R, T, rnd, A)

    return d, side, A


def single_interface_check(r_a, d, ni, nj, tri, Lx, Ly, side, z_cov, pol, wl, Fr_or_TMM, lookuptable):

    decide = {0: decide_RT_Fresnel, 1: decide_RT_TMM}

    #print('initial side: ', side)
    initial_side = side
    # weird stuff happens around edges; can get transmission counted as reflection
    #d = (r_b - r_a) / np.linalg.norm(r_b - r_a)
    intersect = True
    checked_translation = False
    #r_a0 = r_a
    d0 = d

    # ignore the possibility of absorption in the interfaces for now

    # [top, right, bottom, left]
    translation = np.array([[0, -Ly, 0], [-Lx, 0, 0], [0, Ly, 0], [Lx, 0, 0]])
    i1 = 0
    while intersect:
        i1 = i1+1
        #print(i1)
        result = check_intersect(r_a, d, tri)
        if result == False and not checked_translation:

            which_side, tt = exit_side(r_a, d, Lx, Ly)

            r_a = r_a + translation[which_side]
            #ray_path = np.vstack((ray_path, np.append(r_a,1)))
            checked_translation = True

        elif result == False and checked_translation:
            # end, do nothing
            #print(d)
            o_t = np.real(acos(d[2] / (np.linalg.norm(d) ** 2)))
            o_p = np.real(atan2(d[1], d[0]))
            #print('theta_out', o_t*180/np.pi)

            # if you hit an edge, sometimes code will think transmission has happened/
            # try to apply translation, but direction of the ray will be such that
            # the ray is on the other side of the surface than it 'thinks'
            #else:
            if side == initial_side:

                intersect = False  # to stop the while loop
                #print('no more intersections - reflection')
                #print('d_z_i, d_z_end', d0[2], d[2])
                #ray_path = np.vstack((ray_path, np.append(r_a+2*d, 0)))
                final_res = 0
            else:
                #print('no more intersections - transmission')
                #print('d_z_i, d_z_end', d0[2], d[2])
                #ray_path = np.vstack((ray_path, np.append(r_a + 2 * d, 0)))
                intersect = False
                final_res = 1
                # print(intersect, checked_translation)
                # ray travels into next medium

        else:
            # there has been an intersection
            intersn = result[0] # coordinate of the intersection (3D)

            theta =  result[1]
            N = result[2]*side # so angles get worked out correctly, relative to incident face normal

            if side == 1:
                n0 = ni
                n1 = nj

            else:
                n0 = nj
                n1 = ni

            rnd = random()

            d, side, A = decide[Fr_or_TMM](n0, n1, theta, d, N, side, pol, rnd, wl, lookuptable)

            r_a = np.real(intersn + d / 1e9) # this is to make sure the raytracer doesn't immediately just find the same intersection again

            checked_translation = False # reset, need to be able to translate the ray back into the unit cell again if necessary
            #print(intersn)
            #ray_path = np.vstack((ray_path, np.append(intersn, 0)))

            if A is not None:
                intersect = False
                final_res = 2
                o_t = A
                o_p = 0

        # if we are above surface, going downwards, or below surface, going upwards, and we have not yet
        # reached z_cov, we should keep trying to translate, so want to set check_translation = false
        # we also need to update r_a so that check_translation can correctly identify the next unit cell
        # the ray will move into

        if (side == 1 and d[2] < 0 and r_a[2] > z_cov) or \
                (side == -1 and d[2] > 0 and r_a[2] < z_cov) and checked_translation:  # going down but above surface
            exit, t = exit_side(r_a, d, Lx, Ly)
            r_a = r_a + t*d + translation[exit]
            #ray_path = np.vstack((ray_path, np.append(r_a, 1)))
            checked_translation = True

    return final_res, o_t, o_p, r_a, d, theta # theta is LOCAL incidence angle (relative to texture)

def check_intersect(r_a, d, tri):

    # all the stuff which is only surface-dependent (and not dependent on incoming direction) is
    # in the surface object tri.
    pref = 1 / np.sum(np.matlib.repmat(np.transpose([-d]), 1, tri.size).T * tri.crossP, axis=1)
    corner = r_a - tri.P_0s
    t = pref * np.sum(tri.crossP * corner, axis=1)
    u = pref * np.sum(np.cross(tri.P_2s - tri.P_0s, np.matlib.repmat(np.transpose([-d]), 1, tri.size).T) * corner, axis=1)
    v = pref * np.sum(np.cross(np.matlib.repmat(np.transpose([-d]), 1, tri.size).T, tri.P_1s - tri.P_0s) * corner, axis=1)
    As = np.vstack((t, u, v))

    which_intersect = (As[1, :] + As[2, :] <= 1) & (np.all(As[[1, 2],] >= -1e-10, axis=0)) & (As[0, :] > 0) # get errors if set exactly to zero.
    if sum(which_intersect) > 0:

        t = t[which_intersect]
        P0 = tri.P_0s[which_intersect]
        P1 = tri.P_1s[which_intersect]
        P2 = tri.P_2s[which_intersect]
        ind = np.argmin(t)
        t = min(t)

        # print('Intersection!')
        intersn = r_a + t * d
        N = np.cross(P1[ind] - P0[ind], P2[ind] - P0[ind])
        N = N / np.linalg.norm(N)
        # only need the face normal of the relevant triangle

        theta = atan(np.linalg.norm(np.cross(N, -d))/np.dot(N, -d))  # in radians, angle relative to plane
        #theta = atan2(np.linalg.norm(np.cross(N, -d)),np.dot(N, -d)) # in radians, angle relative to plane
        # this is negative if approaching from below


        return [intersn, theta, N]

    else:

        return False


