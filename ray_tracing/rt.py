import numpy as np
import numpy.matlib
from scipy.spatial import Delaunay
from cmath import sin, cos, asin, sqrt, acos, atan
from math import atan2, ceil
from random import random

# want a Delaunay-type triangulation object which has the same attributes as
# the return from scipy.spatial.Delaunaytheta_t = np.asin((n1/n2)*np.sin(theta)), but contains 3D rather than 2D coordinates,
# i.e. a 2D surface in 3D space.




def RT(group, incidence, transmission, options):
    wavelengths = options['wavelengths']

    mats = [incidence]
    for i1 in range(len(group.materials)):
        mats.append(group.materials[i1])
    mats.append(transmission)

    theta = options['theta']
    phi = options['phi']
    surfaces = group.interfaces

    I_thresh = options['I_thresh']

    widths =  group.widths
    widths.insert(0, 0)
    widths.append(0)
    widths = 1e6*np.array(widths)  # convert to um

    z_space = 1e6*group.depth_spacing
    z_pos = np.arange(0, sum(widths), z_space)

    nks = np.empty((len(mats), len(wavelengths)), dtype=complex)
    R = np.zeros(len(wavelengths))
    T = np.zeros(len(wavelengths))

    absorption_profiles = np.zeros((len(wavelengths), len(z_pos)))
    A_layer = np.zeros((len(wavelengths), len(widths)))

    for i1, mat in enumerate(mats):
        nks[i1] = mat.n(wavelengths) + 1j*mat.k(wavelengths)

    h = max(surfaces[0].Points[:, 2])
    r = abs((h + 1) / cos(theta))
    r_a_0 = np.real(np.array([r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta)]))

    x_lim = surfaces[0].Lx
    y_lim = surfaces[0].Ly

    nx = options['nx']
    ny = options['ny']

    xs = np.linspace(x_lim/100, x_lim-(x_lim/100), nx)
    ys = np.linspace(y_lim/100, y_lim-(y_lim/100), ny)

    xv, yv = np.meshgrid(xs, ys)
    xv = xv.flatten()
    yv = yv.flatten()

    # need to calculate r_a and r_b

    for i1, wl in enumerate(wavelengths):
        print(i1)
        materials = {'nk': nks[:,i1], 'alpha': np.imag(nks[:,i1])*4*np.pi/(wl*1e6), 'width': widths}

        for c, value in enumerate(xv):

            I, profile, A_per_layer, th_o, ray_path = single_ray(xv[c], yv[c], r_a_0, theta, phi, surfaces, materials, z_pos, I_thresh)
            absorption_profiles[i1] = absorption_profiles[i1] + profile/(nx*ny)
            A_layer[i1] = A_layer[i1] + A_per_layer/(nx*ny)
            if th_o is not None:
                if np.real(th_o) < np.pi/2:
                    R[i1] = R[i1] + I/(nx*ny)
                else:
                    T[i1] = T[i1] + I/(nx*ny)



    return R, T, A_layer, absorption_profiles



class RTSurface:
    def __init__(self, Points):
        # want the 'base' of the scan to lie on the plane z = 0
        # so if min(Points(:, 3) < 0 or > 0, translate

        # Points[:, 2] = Points[:, 2] - min(Points[:,2])
        tri = Delaunay(Points[:, [0,1]])
        self.simplices = tri.simplices
        self.Points = Points
        self.P_0s = Points[tri.simplices[:, 0]]
        self.P_1s = Points[tri.simplices[:, 1]]
        self.P_2s = Points[tri.simplices[:, 2]]
        self.crossP = np.cross(self.P_1s - self.P_0s, self.P_2s - self.P_0s)
        self.size = self.P_0s.shape[0]
        self.Lx = abs(min(Points[:,0])-max(Points[:,0]))
        self.Ly = abs(min(Points[:,1])-max(Points[:,1]))

        self.zcov= Points[:,2][np.all(np.array([Points[:,0] == min(Points[:,0]), Points[:,1] == min(Points[:,1])]), axis = 0)]
        #z1 = Points[:,2][np.all(np.array([Points[:,0] == min(Points[:,0]), Points[:,1] == max(Points[:,1])]), axis = 0)]
        #z2 = Points[:,2][np.all(np.array([Points[:,0] == max(Points[:,0]), Points[:,1] == min(Points[:,1])]), axis = 0)]
        #z3 = Points[:,2][np.all(np.array([Points[:,0] == max(Points[:,0]), Points[:,1] == max(Points[:,1])]), axis = 0)]



def calc_R(n1, n2, theta):
    theta_t = np.arcsin((n1/n2)*np.sin(theta))
    Rs = np.abs((n1*np.cos(theta)-n2*np.cos(theta_t))/(n1*np.cos(theta)+n2*np.cos(theta_t)))**2
    Rp = np.abs((n1 * np.cos(theta_t) - n2 * np.cos(theta)) / (n1 * np.cos(theta_t) + n2 * np.cos(theta)))**2
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


def single_ray(x, y, r_a_0, theta, phi, surfaces, materials, z_pos, I_thresh):
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
    widths = materials['width']
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
    ray_path = np.append(r_a, 0)
    n_passes = 0

    depths = []
    depth_indices = []
    for i1 in range(len(widths)):
        depth_indices.append((z_pos < np.cumsum(widths)[i1]) & (z_pos >= np.cumsum(widths)[i1 - 1]))
        depths.append(z_pos[depth_indices[i1]] - \
             np.cumsum(widths)[i1 - 1])


    while not stop:
        #print('surface: ', surf_index)
        #print('material: ', mat_index)
        #print('direction:', direction)
        #ax.plot([r_a[0], r_a[0] + d[0]], [r_a[1], r_a[1] + d[1]], [r_a[2], r_a[2] + d[2]])
        # translate r_a so that the ray can actually intersect with the unit cell of the surface it is approaching
        surf = surfaces[surf_index]
        #print(r_a, d)
        #print('translating...')
        r_a[0] = r_a[0]-surf.Lx*((r_a[0]+d[0]*(surf.zcov-r_a[2])/d[2])//surf.Lx)
        r_a[1] = r_a[1]-surf.Ly*((r_a[1]+d[1]*(surf.zcov-r_a[2])/d[2])//surf.Ly)

        ray_path = np.vstack((ray_path, np.append(r_a + d*(((surf.zcov - r_a[2])/d[2])-2), 1)))
        #print(r_a, d)
        res, theta, r_a, d, ray_path = single_interface_check(r_a, d, materials['nk'][mat_index],
                                     materials['nk'][mat_index+direction], surf, surf.Lx, surf.Ly, direction, surf.zcov, ray_path)

        if res == 0:  # reflection
            direction = -direction # changing direction due to reflection

            # staying in the same material, so mat_index does not change, but surf_index does
            surf_index = surf_index + direction

        if res == 1:  # transmission
            surf_index = surf_index + direction
            mat_index = mat_index + direction # is this right?

        #print(materials['width'][mat_index])
        #print(materials['alpha'][mat_index])
        #print(depths[mat_index])
        I_b = I
        #print(mat_index)
        DA, stop, I, theta = traverse(widths[mat_index], theta, materials['alpha'][mat_index], x, y, I,
                                      depths[mat_index], I_thresh, direction)
        A_per_layer[mat_index] = A_per_layer[mat_index] + I_b - I
        profile[depth_indices[mat_index]] = profile[depth_indices[mat_index]] + DA

        n_passes = n_passes + 1
        #print(DA)
        # theta = None means absorbed

        if direction == 1 and mat_index == (len(materials['width'])-1):
            stop = True
            # have ended with transmission

        elif direction == -1 and mat_index == 0:
            stop = True
            # have ended with reflection
            # final results to pass are
        #print('stop', stop)

    #print('n_passes', n_passes)
    return I, profile, A_per_layer, theta, ray_path



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




def single_interface_check(r_a, d, n0, n1, tri, Lx, Ly, side, z_cov, ray_path):
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
            # do translation
            #print('no intersection/not tried translation')

            #dir2D = calc_angle(d[[0,1]])
            # if sum(r_a[0:2] >= 1):
            #     print('translation')
            #     print(r_a, d)
            #     print(ray_path)
            #     print(r_b)

            #print(Lx, Ly, r_a)
            #angles = [calc_angle(x) for x in [[Lx - r_a[0], Ly - r_a[1]], [Lx - r_a[0], -r_a[1]], [-r_a[0], -r_a[1]], [-r_a[0], Ly - r_a[1]]]]
            #angles = [2 * np.pi + x if x < 0 else x for x in np.append(dir2D, angles)]
            #print(angles)
            #which_side = np.digitize([angles[0]], angles[1:])[0] % 4  # modulu division because want 4 -> 0

            #n = np.array([[0, -1, 0], [-1, 0, 0], [0, 1, 0], [1, 0, 0]])
            #p_0 = np.array([[0, Ly, 0], [Lx, 0, 0], [0, 0, 0], [0, 0, 0]])
            #t = np.sum((p_0 - r_a) * n, axis=1) / np.sum(d * n, axis=1)
            #which_intersect = t > 0
            #t[~which_intersect] = float('inf')
            which_side, tt = exit_side(r_a, d, Lx, Ly)
            #print('which side, translation:', which_side, translation[which_side])

            #print(which_side)
            # if the ray is exiting at the top, want to translate its starting point (intersn) down (minus) by Ly
            # if the ray is exiting at the right, want to translate its starting point left (minus) by Lx... etc.
            #print('r_a before translation', r_a)
            r_a = r_a + translation[which_side]
            ray_path = np.vstack((ray_path, np.append(r_a,1)))
            checked_translation = True
            #print('applied translation')
            #print('r_a, d: ', r_a, d)

            #print(r_a, d)
            #print(ray_path)


        elif result == False and checked_translation:
            # end, do nothing
            #print(d)
            o_t = acos(d[2] / (np.linalg.norm(d) ** 2))
            #print('theta_out', o_t)


            #if (side == 1 and o_t > np.pi/2) or (side == -1 and o_t < np.pi/2):
                #print('something weird happened')
                #print('r_a, d, theta, side: ', r_a, d, o_t, side)
                #r_a = r_a0 + 0.0001*np.array([random(), random(), 0])
                #d = d0
                #checked_translation = False
                #side = initial_side
                #print(ray_path)
                # #final_res = 3

            # if you hit an edge, sometimes code will think transmission has happened/
            # try to apply translation, but direction of the ray will be such that
            # the ray is on the other side of the surface than it 'thinks'
            #else:
            if side == initial_side:

                intersect = False  # to stop the while loop
                #print('no more intersections - reflection')
                #print('d_z_i, d_z_end', d0[2], d[2])
                ray_path = np.vstack((ray_path, np.append(r_a+2*d, 0)))
                final_res = 0
            else:
                #print('no more intersections - transmission')
                #print('d_z_i, d_z_end', d0[2], d[2])
                ray_path = np.vstack((ray_path, np.append(r_a + 2 * d, 0)))
                intersect = False
                final_res = 1
                # print(intersect, checked_translation)
                # ray travels into next medium


        else:
            # there has been an intersection
            intersn = result[0] # coordinate of the intersection (3D)
            #print('intersection:', intersn)
            theta =  result[1]
            N = result[2]*side # so angles get worked out correctly, relative to incident face normal
            #print('intersn, theta, N', result)

            #ax.plot([intersn[0], intersn[0]+N[0]], [intersn[1], intersn[1]+N[1]], [intersn[2], intersn[2]+N[2]])

            if side == 1:
                ni = n0
                nj = n1
            else:
                ni = n1
                nj = n0

            #print('ni, nj', ni, nj)

            #print(nj, ni)


            rnd = random()
            #print(theta)
            R = calc_R(ni, nj, theta)
            #print('calculated R', R)

            if rnd <= R: # REFLECTION
                # reflection, theta_out = theta_in
                d = np.real(d - 2 * np.dot(d, N) * N)
                d = d/np.linalg.norm(d)
                #print('R')

            else: # TRANSMISSION
                    # transmission, refraction
                    # for now, ignore effect of k on refraction
                #print('d before', d)
                tr_par = ((np.real(n0)/np.real(n1))**side)*(d-np.dot(d, N)*N)
                tr_perp = -sqrt(1-np.linalg.norm(tr_par)**2)*N
                #side = -side  set side explicitly, by checking, to avoid problems with hitting edges
                d = np.real(tr_par+tr_perp)
                d = d/np.linalg.norm(d)
                #print('d after', d)
                #print('T')


            r_a = np.real(intersn + d / 1e9) # this is to make sure the raytracer doesn't immediately just find the same intersection again
            if r_a[2] < intersn[2]:
                #print('below surface', side)
                side = -1
            else:
                #print('above surface', side)
                side = 1
            checked_translation = False # reset, need to be able to translate the ray back into the unit cell again if necessary
            #print(intersn)
            ray_path = np.vstack((ray_path, np.append(intersn, 0)))

            #ax.plot(ray_path[:, 0], ray_path[:, 1], ray_path[:, 2])

        # if we are above surface, going downwards, or below surface, going upwards, and we have not yet
        # reached z_cov, we should keep trying to translate, so want to set check_translation = false
        # we also need to update r_a so that check_translation can correctly identify the next unit cell
        # the ray will move into

        if (side == 1 and d[2] < 0 and r_a[2] > z_cov) or \
                (side == -1 and d[2] > 0 and r_a[2] < z_cov):  # going down but above surface
            exit, t = exit_side(r_a, d, Lx, Ly)

            r_a = r_a + t*d + translation[exit]
            ray_path = np.vstack((ray_path, np.append(r_a, 1)))
            checked_translation = True
            #print('extra translation, r_a / d / z_cov / t = ', r_a, d, z_cov, t)



    # final result we need is theta and phi out:
    #ax.plot([r_a[0], r_a[0] + d[0]], [r_a[1], r_a[1] + d[1]], [r_a[2], r_a[2] + d[2]])
    return final_res, o_t, r_a, d, ray_path

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

        theta = abs(atan(np.linalg.norm(np.cross(N, -d))/np.dot(N, -d))) # in radians, angle relative to plane
        # this is negative if approaching from below


        return [intersn, theta, N]

    else:
        # print('As', As[1, :] + As[2, :])
        # print('cond1',  As[1, :] + As[2, :] <= 1)
        # print(As[[1, 2],])
        # print('cond2', As[[1, 2],] >= -1e-16) # get errors if set exactly to zero.
        # print('cond2', np.all(As[[1, 2],] >= -1e-16, axis=0))
        # print(As[0, :])
        # print('cond3', As[0, :] > 0)
        # print((As[1, :] + As[2, :] <= 1) & (np.all(As[[1, 2],] >= 0, axis=0)) & (As[0, :] > 0))
        return False


