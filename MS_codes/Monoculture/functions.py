import numpy as np
from PIL import Image
import time
import copy
#from numba import jit, f8
#from numba.types import UniTuple

def resolve_down(im, factor):
    """averages value over a square region of pixels of size factor x factor"""

    # get pixel size of image
    Lx, Ly = im.size

    # get pixel offset towards all 4 margins and crop them out
    LR_off = Lx % factor
    L_off = int(LR_off / 2)
    R_off = Lx - LR_off + L_off
    TB_off = Ly % factor
    T_off = int(TB_off / 2)
    B_off = Ly - TB_off + T_off

    im1 = np.array(im.crop((L_off, T_off, R_off, B_off)))

    # resolve down image along the rows
    im2 = []
    for a in im1:
        b = a.reshape(-1, factor).mean(axis=1)
        im2.append(b)
    im2 = np.array(im2).T

    # resolve down the image along the columns
    im3 = []
    for a in im2:
        b = a.reshape(-1, factor).mean(axis=1)
        im3.append(b)
    im3 = np.array(im3).T

    return im3

def read_data_as_array(input):
    """ for given file location, time-series data of Phase and fluorescence channel are imported as numpy arrays """

    # Read input parameters for data formatting
    path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min = tuple(input)

    Phase = []
    FC = []

    t = time_i+1
    while t < time_f:
        imp = Image.open(path + str(t).zfill(2) + "_Phase.jpg").convert('L')
        imp_resized = resolve_down(imp, nump)
        imf = Image.open(path + str(t).zfill(2) + "_{}.jpg".format(fc)).convert('L')
        imf_resized = resolve_down(imf, nump)

        Phase.append(imp_resized)
        FC.append(imf_resized)
        t += 1

    Phase = np.array(Phase)
    FC = np.array(FC)

    """ Use fluorescence channel as a mask to eliminate intensities in 'cell-free' regions of phase channel images """
    """ Using a threshold to account for dispersion of emitted light """
    for i in range(len(Phase)):
        cut = thresh * np.max(FC[i])
        FC[i][FC[i] <= cut] = 0
        Phase[i][FC[i] <= cut] = 0

    return(Phase, FC)

def nbr_sum(M):
    """sum of neighboring values of elements of matrix M"""
    L = np.roll(M, (0, -1), (0, 1))   # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L

#@jit(f8[:,:](f8[:,:]), nopython=True)
def nbr_sum_jit(M):
    L = np.zeros(M.shape)
    for y in range(M.shape[1]-1):
        L[:, y] = M[:, y+1] + M[:, y-1]
    for x in range(M.shape[0]-1):
        L[x, :] = M[x+1, :] + M[x-1, :]
    return L

#@jit(f8[:,:](f8[:,:]), nopython=True)
def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += nbr_sum(M)

    return L

def divide(M1, M2):
    """To divide two numpy arrays M1/M2 element-wise and avoid division by zero"""
    f = np.divide(M1, M2, out=np.zeros_like(M1), where=M2 != 0)
    return f

#@jit(f8[:,:](f8[:,:], f8), nopython=True)
def hill(A, Ahf):
    return A/(Ahf+A)

#@jit(f8[:,:](f8[:,:], f8), nopython=True)
def inv_hill(A, Ahf):
    return Ahf/(Ahf+A)

#@jit(UniTuple(f8[:,:],2)(f8[:,:],f8[:,:],f8[:],f8, f8), nopython=True)
def model_update(C, S, Parameters, res, dt):
    """Update state variables according to the single-species continuum model"""
    k, mu, Rhf, Qhf, Phf, Y, Ks, DC, DS = Parameters

    R, Q, P = inv_hill(C, Rhf), hill(C, Qhf), inv_hill(C, Phf)

    # maximum growth rate
    M = mu * C * hill(S, Ks)

    Sdiff = res * DS * discrete_laplacian(S) - dt * M * R / Y
    Cdiff = res * DC * (P * nbr_sum(M * Q) - M * Q * nbr_sum(P)) + dt * M * R

    C += Cdiff
    S += Sdiff

    # Remove negative values
    S[S < 0] = 0
    C[C < 0] = 0
    # Reflecting boundaries
    C[0, :], C[-1, :], C[:, 0], C[:, -1], S[0, :], S[-1, :], S[:, 0], S[:, -1] \
        = C[1, :], C[-2, :], C[:, 1], C[:, -2], S[1, :], S[-2, :], S[:, 1], S[:, -2]

    return C, S

def check_bounds(Parameters, l_Parameters, u_Parameters):
    """Return 1 if Parameters are outside specified boundaries and 0 if within boundaries"""
    length = len(Parameters)
    check = 0
    ind = 0
    while ind < length:
        if Parameters[ind] >= u_Parameters[ind]:
            check = 1
        elif Parameters[ind] <= l_Parameters[ind]:
            check = 1
        ind += 1
    return(check)


def get_error(Phase, FC, Parameters, l_Parameters, u_Parameters, input):
    """Return error/cost function value for given parameters and dataset"""

    if check_bounds(Parameters, l_Parameters, u_Parameters) > 0:
        print("Move out of bounds")
        error = 1e50
    else:
        # Read input parameters
        path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min = input

        # Time Resolution in sec
        dt = 1
        # Space resolution in um
        dx = nump * umpp

        # Time steps
        T = int((time_f - 1) * dthr * 3600 - time_i)
        res = dt / dx ** 2

        # Initialize
        C = copy.deepcopy(FC[0])*Parameters[0]
        S = np.ones(C.shape)

        # Simulate
        update_every = dthr * 3600  # number of time steps after which data is stored
        C_time = []
        S_time = []

        start = time.time()
        for tt in range(T):
            if tt % update_every == 0:
                C_time.append(C.copy())
                S_time.append(S.copy())
            C, S = model_update(C, S, Parameters, res, dt)
        end = time.time()

        print("Model run time = {}".format(end-start))

        """ computing error """
        error = 0
        t = time_i
        while t < time_f - 2:
            sd = 0.1*Phase[t]
            sd[sd < sd_min] = sd_min
            error += np.sum(divide((C_time[t] - Phase[t]) ** 2, sd**2))
            t += 1


    return (error)

def get_error_NB(Phase, FC, Parameters, input):
    """Return error/cost function value for given parameters and dataset"""

    # Read input parameters
    path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min = tuple(input)

    # Time Resolution in sec
    dt = 1
    # Space resolution in um
    dx = nump * umpp

    # Time steps
    T = int((time_f - 1) * dthr * 3600 - time_i)
    res = dt / dx ** 2

    # Initialize
    k = Parameters[0]
    C = copy.deepcopy(FC[0])*k
    C[C < 0] = 0
    S = np.ones(C.shape)

    # Simulate
    update_every = dthr * 3600  # number of time steps after which data is stored
    C_time = []
    S_time = []

    for tt in range(T):
        if tt % update_every == 0:
            C_time.append(C.copy())
            S_time.append(S.copy())
        C, S = model_update(C, S, Parameters, res, dt)

    """ computing error """
    error = 0
    t = time_i
    while t < time_f - 2:
        sd = 0.1*Phase[t]
        sd[sd < sd_min] = sd_min
        error += np.sum(divide((C_time[t] - Phase[t]) ** 2, sd**2))
        t += 1


    return (error)


