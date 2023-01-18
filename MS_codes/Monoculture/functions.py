import numpy as np
from PIL import Image
from math import *
import time
import copy

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
    new = []
    for a in im1:
        b = a.reshape(-1, factor).mean(axis=1)
        new.append(b)
    new = np.array(new).T

    # resolve down the image along the columns
    neww = []
    for a in new:
        b = a.reshape(-1, factor).mean(axis=1)
        neww.append(b)
    newww = np.array(neww).T

    return newww

def read_data_as_array(input):
    """ for given file location, time-series data of Phase and fluorescence channel are imported as numpy arrays """

    # Read input parameters for data formatting
    path, fc, time_i, time_f, nump, umpp, thresh_ph, thresh_fl, dthr = tuple(input)

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

    Phase = np.array(Phase) - thresh_ph
    FC = np.array(FC) - thresh_fl
    Phase[Phase < 0] = 0
    FC[FC < 0] = 0

    return(Phase, FC)


def nbr_sum(M):
    """sum of neighboring values of elements of matrix M"""
    L = np.roll(M, (0, -1), (0, 1))   # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L


def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += nbr_sum(M)

    return L


def divide(M1, M2):
    """To divide two numpy arrays M1/M2 element-wise and avoid division by zero"""
    f = np.divide(M1, M2, out=np.zeros_like(M1), where=M2 != 0)
    return f


def model_update(C, S, Parameters, res, dt):
    """Update state variables according to the single-species continuum model"""
    mu, Rhf, Qhf, Phf, Y, log10Ks, DC, DS = tuple(Parameters)

    R = C / (Rhf + C)
    Q = C / (Qhf + C)
    P = Phf / (Phf + C)
    Ks = 10 ** log10Ks

    Sdiff = res * DS * discrete_laplacian(S) - dt * mu * C * divide(S, Y * (Ks + S))
    Cdiff = dt * mu * (1 - R) * C * S / (Ks + S) + res * DC * mu * (
                P * nbr_sum(C * Q * S / (Ks + S)) - Q * C * S * nbr_sum(P) / (Ks + S))

    C += Cdiff
    S += Sdiff
    # Remove negative values
    S[S < 0] = 0

    # Reflecting boundaries
    C[0, :], S[0, :] = C[1, :], S[1, :]  # top
    C[-1, :], S[-1, :] = C[-2, :], S[-2, :]  # bottom
    C[:, 0], S[:, 0] = C[:, 1], S[:, 1]  # left
    C[:, -1], S[:, -1] = C[:, -2], S[:, -2]  # right

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
        path, fc, time_i, time_f, nump, umpp, thresh_ph, thresh_fl, dthr = tuple(input)

        # Time Resolution in sec
        dt = 1
        # Space resolution in um
        dx = nump * umpp

        # Time steps
        T = int((time_f - 1) * dthr * 3600 - time_i)
        res = dt / dx ** 2

        # Initialize
        k = 10**Parameters[0]
        C = copy.deepcopy(FC[0])*k
        C[C < 0] = 0
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
        error = 0

        t = time_i

        while t < time_f - 2:

            error += np.sum((C_time[t] - Phase[t]) ** 2)
            t += 1


    return (sqrt(error))

