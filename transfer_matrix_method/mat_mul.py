import numpy as np
import xarray as xr
from sparse import load_npz, dot, tensordot, COO, stack
from config import results_path
from angles import make_angle_vector
from solcore import material
import os
import matplotlib.pyplot as plt

lookuptable = xr.open_dataset('C://Users//pmpea//Box Sync//Optics package//results//testing//GaAsGaAsstack.nc')

theta_intv, phi_intv, angle_vector = make_angle_vector(100, np.pi/2, 0.25)
n_a_in = int(len(angle_vector)/2)

Si = material('Si')()
num_wl = 20

wls = np.linspace(700, 1100, num_wl)*1e-9
alphas = Si.alpha(wls)

# bulk thickness in m
thick = 1000*1e-6

thetas = angle_vector[:n_a_in, 1]

def make_D(alphas, thick, thetas):
    diag = np.exp(-alphas[:, None] * thick / np.cos(thetas[None, :]))
    D_1 = stack([COO(np.diag(x)) for x in diag])
    return D_1

D_1 = make_D(alphas, thick, thetas)

v0 = np.zeros((num_wl, n_a_in))
v0[:,0] = 1

mat_path = os.path.join(results_path, 'testing', 'GaAsGaAsstackfrontRT.npz')
absmat_path = os.path.join(results_path, 'testing', 'GaAsGaAsstackfrontA.npz')

fullmat = load_npz(mat_path)
absmat = load_npz(absmat_path)

Rf_1 = fullmat[:, :n_a_in, :]
Tf_1 = fullmat[:, n_a_in:, :]
Af_1 = absmat

Rf_2 = fullmat[:, :n_a_in,:]
Tf_2 = fullmat[:, n_a_in:, :]
Af_2 = absmat

a = np.sum(Rf_2[0].todense(), 0) + np.sum(Tf_2[0].todense(), 0) + np.sum(Af_2[0].todense(), 0)
print(np.all(np.abs(a-1) < 1e-9)) # rounding errors

mat_path = os.path.join(results_path, 'testing', 'GaAsGaAsstackrearRT.npz')
absmat_path = os.path.join(results_path, 'testing', 'GaAsGaAsstackrearA.npz')

fullmat = load_npz(mat_path)
absmat = load_npz(absmat_path)

Rb_1 = fullmat[:, :n_a_in,:]
Tb_1 = fullmat[:, n_a_in:, :]
Ab_1 = absmat

print(np.sum(Rf_1[0].todense(), 0))
print(np.sum(Tf_1[0].todense(), 0))
print(np.sum(Af_1[0].todense(), 0))

a = np.sum(Rf_1[0].todense(), 0) + np.sum(Tf_1[0].todense(), 0) + np.sum(Af_1[0].todense(), 0)
print(np.all(np.abs(a-1) < 1e-9)) # rounding errors

# v0.groupby('wl').apply(lambda x: dot(R, x.data))

def dot_wl(mat, vec):
    result = np.empty((vec.shape[0], mat.shape[1]))
    for i1 in range(vec.shape[0]):
        result[i1, :] = dot(mat[i1], vec[i1])
    return result

n_layers = 2

a_1 = []
vr_1 = []
a_2 = []
vt_2 = []
A_1 = []

vf_1 = dot_wl(Tf_1, v0) # pass through front surface
vr_1.append(dot_wl(Rf_1, v0)) # reflected from front surface
a_1.append(dot_wl(Af_1, v0)) # absorbed in front surface at first interaction
power = np.sum(vf_1, axis=1)
# rep
i1=1

while np.any(power > 1e-15):
    print(i1)
    # vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
    # vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    # vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
    # vf_3 = dot_wl(Rb_1, vf_2) # reflect from front surface
    # power = np.sum(vf_3, axis=1)
    vb_1 = dot_wl(D_1, vf_1) # pass through bulk, downwards
    A_1.append(np.sum(vf_1, 1) - np.sum(vb_1, 1))
    vb_2 = dot_wl(Rf_2, vb_1) # reflect from back surface
    vf_2 = dot_wl(D_1, vb_2) # pass through bulk, upwards
    A_1.append(np.sum(vb_2, 1) - np.sum(vf_2, 1))
    vf_1 = dot_wl(Rb_1, vf_2) # reflect from front surface
    power = np.sum(vf_1, axis=1)

    vr_1.append(dot_wl(Tb_1, vf_2))  # matrix travelling up in medium 0, i.e. reflected overall by being transmitted through front surface
    vt_2.append(dot_wl(Tf_2, vb_1))  # transmitted into medium below through back surface
    a_2.append(dot_wl(Af_2, vb_1))  # absorbed in 2nd surface
    a_1.append(dot_wl(Ab_1, vf_2))  # absorbed in 1st surface (from the back)

    i1+=1


a_1 = np.array(a_1)
vr_1 = np.array(vr_1)
a_2 = np.array(a_2)
vt_2 = np.array(vt_2)
A_1 = np.array(A_1)

# other results:

# end rep
R = np.sum(vr_1, (0,2))
T = np.sum(vt_2, (0,2))
A = np.sum(A_1, 0)
A_front = np.sum(a_1, (0,2))
A_back = np.sum(a_2, (0,2))
plt.figure()

plt.plot(wls, R)
plt.plot(wls, A)
plt.plot(wls, A_front)
plt.plot(wls, A_back)
plt.plot(wls, T)
plt.plot(wls, R+A+A_front+A_back+T)
plt.legend(['R', 'A', 'Af', 'Ab', 'T'])

a_l_front = np.sum(a_1, 0)
a_l_back = np.sum(a_2, 0)
plt.figure()
plt.plot(wls, a_l_front[:,0])
plt.plot(wls, a_l_front[:,1])
plt.legend(np.arange(2))

class absorp_surface_fn:
    """

    Absorption in a given layer is a pretty simple analytical function:
    The sum of four exponentials.

    a(z) = A1*exp(a1*z) + A2*exp(-a1*z)
           + A3*exp(1j*a3*z) + conj(A3)*exp(-1j*a3*z)

    where a(z) is absorption at depth z, with z=0 being the start of the layer,
    and A1,A2,a1,a3 are real numbers, with a1>0, a3>0, and A3 is complex.
    The class stores these five parameters, as well as d, the layer thickness.

    This gives absorption as a fraction of intensity coming towards the first
    layer of the stack.
    """

    def __init__(self, arr):
        # (5, 1) array, order of rows: A1, A2, A3, a1, a3
        self.A1 = arr[:,0]
        self.A2 = arr[:,1]
        self.A3 = arr[:,2]
        self.a1 = arr[:,3]
        self.a3 = arr[:,4]

    def run(self, z):
        """
        Calculates absorption at a given depth z, where z=0 is the start of the
        layer.
        """

        part1 = self.A1[:, None] * np.exp(self.a1[:, None] * z[None, :])
        part2 = self.A2[:, None] * np.exp(-self.a1[:, None] * z[None, :])
        part3 = self.A3[:, None] * np.exp(1j * self.a3[:, None] * z[None, :])
        part4 = np.conj(self.A3[:, None]) * np.exp(-1j * self.a3[:, None] * z[None, :])

        part1[self.A1 < 1e-100, :] = 0

        return (part1 + part2 + part3 + part4)/len(self.A1)

    def flip(self):
        """
        Flip the function front-to-back, to describe a(d-z) instead of a(z),
        where d is layer thickness.
        """
        expn = np.exp(self.a1 * self.d)
        #expn[expn > 1e100] = 1e100
        newA1 = self.A2 * np.exp(-self.a1 * self.d)
        newA1[self.A2 == 0] = 0
        newA2 = self.A1 * expn
        newA2[self.A1 == 0] = 0
        self.A1, self.A2 = newA1, newA2
        self.A3 = np.conj(self.A3 * np.exp(1j * self.a3 * self.d))
        return self

    def scale(self, factor):
        """
        multiplies the absorption at each point by "factor".
        """
        self.A1 *= factor
        self.A2 *= factor
        self.A3 *= factor
        self.A1[np.isnan(self.A1)] = 0
        self.A2[np.isnan(self.A2)] = 0
        self.A3[np.isnan(self.A3)] = 0
        return self
