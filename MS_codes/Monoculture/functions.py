import numpy as np
from PIL import Image
import time
import copy
import matplotlib.pyplot as plt
from math import sqrt
#from numba import jit, f8
#from numba.types import UniTuple

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))

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

def resolve_down_get_std(im, factor):
    """averages value over a square region of pixels of size (factor, factor),
    and get stand deviation of values in each of these square regions"""

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

    # get standard deviation of values in each grid to be compressed
    std_im = []
    A = blockshaped(im1, factor, factor)
    for i in A:
        std_im.append(np.std(i))
    std_im = np.reshape(std_im, (-1, (Lx-LR_off)//factor))

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

    return im3, std_im

def read_data_as_array(input):
    """ for given file location, time-series data of Phase and fluorescence channel are imported as numpy arrays """

    # Read input parameters for data formatting
    path, fc, time_i, time_f, nump, umpp, thresh, dthr = input

    Phase = []
    FC = []
    stds = []
    t = time_i+1
    while t < time_f+1:
        imp = Image.open(path + str(t).zfill(2) + "_Phase.jpg").convert('L')
        imp_resized, stds_t = resolve_down_get_std(imp, nump)
        imf = Image.open(path + str(t).zfill(2) + "_{}.jpg".format(fc)).convert('L')
        imf_resized = resolve_down(imf, nump)

        Phase.append(imp_resized)
        FC.append(imf_resized)
        stds.append(stds_t)
        t += 1

    Phase = np.array(Phase)
    FC = np.array(FC)
    stds = np.array(stds)

    # get list of intensities in cell-free regions from t=0 image
    cut = thresh * np.max(FC[0])
    BG = []
    for i in range(len(Phase[0])):
        for j in range(len(Phase[0][0])):
            if FC[0][i][j] <= cut:
                BG.append(Phase[0][i][j])

    """ Use fluorescence channel as a mask to eliminate intensities in 'cell-free' regions of phase channel images """
    """ Using a threshold to account for dispersion of emitted light """
    for i in range(len(Phase)):
        cut = thresh * np.max(FC[i])
        FC[i][FC[i] <= cut] = 0
        Phase[i][FC[i] <= cut] = 0
        stds[i][FC[i] <= cut] = 0

    return(Phase, FC, BG, stds)


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
    """Return 1 if at least one parameter is outside specified boundaries and 0 if all are within boundaries"""
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

def compute_error(data, model, stds, BG_stat):
    """Compute net error due to convolution of background noise and dynamic noise"""

    mean_BG, std_BG = BG_stat

    sse = np.zeros(data[0].shape)
    inv_var_pdt = np.ones(data[0].shape)*(1/std_BG**2)
    mean_pdt = np.ones(data[0].shape)*(mean_BG/std_BG**2)
    time_term = np.zeros(data[0].shape)

    for t in range(len(data)):
        check = np.ones(data[t].shape)
        check[data[t] == 0] = 0
        time_term += check
        sse += divide((data[t] - model[t]) ** 2, stds[t] ** 2)
        inv_var_pdt += divide(np.ones(data[t].shape),stds[t]**2)
        mean_pdt += divide(data[t] - model[t], stds[t] ** 2)
        t += 1

    var_pdt = divide(np.ones(data[0].shape), inv_var_pdt)
    mean_pdt *= var_pdt

    pdt_term = divide(mean_pdt**2, var_pdt)
    pdt_term[time_term==0] = 0

    error = np.sum(sse - pdt_term + time_term*1.8379)

    return(error)

def get_error(Phase, init, BG_stat, stds, Parameters, l_Parameters, u_Parameters, input):
    """Return error/cost function value for given parameters and dataset"""

    if check_bounds(Parameters, l_Parameters, u_Parameters) > 0:
        print("Move out of bounds")
        error = 1e50
    else:
        # Read input parameters
        path, fc, time_i, time_f, nump, umpp, thresh, dthr = input

        # Time Resolution in sec
        dt = 1
        # Space resolution in um
        dx = nump * umpp

        # Time steps
        T = int(time_f * dthr * 3600 - time_i)
        res = dt / dx ** 2

        # Initialize
        C = copy.deepcopy(init)*Parameters[0]
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

        error = compute_error(Phase, C_time, stds, BG_stat)

    return (error)

def get_error_NB(Phase, FC, BG_stat, stds, Parameters, input):
    """Return error/cost function value for given parameters and dataset"""

    # Read input parameters
    path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min = tuple(input)

    # Time Resolution in sec
    dt = 1
    # Space resolution in um
    dx = nump * umpp

    # Time steps
    T = int(time_f * dthr * 3600 - time_i)
    res = dt / dx ** 2

    # Initialize
    k = Parameters[0]
    C = copy.deepcopy(FC[0])*k
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

    error = compute_error(Phase, C_time, stds, BG_stat)


    return (error)


def get_sd(Phase, FC, input):
    path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min = input

    stds = []
    t = time_i
    while t < time_f:
        Phase_raw = Image.open(path + str(t+1).zfill(2) + "_Phase.jpg").convert('L')
        std_t = []
        A = blockshaped(Phase_raw, nump, nump)
        for i in A:
            std_t.append(np.std(i))
        B = np.reshape(std_t, (-1, 101))
        stds.append(B)
        t += 1
    stds = np.array(stds)
    #stds = np.sqrt(stds**2 + 0.0625*Phase**2)

    for i in range(len(Phase)):
        cut = 0.1 * np.max(FC[i])
        stds[i][FC[i] <= cut] = 0

    return(stds)

def get_error_given_sd(M, sd, init, Parameters, l_Parameters, u_Parameters, input):
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
        T = int(time_f * dthr * 3600 - time_i)
        res = dt / dx ** 2

        # Initialize
        C = copy.deepcopy(init)*Parameters[0]
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
        while t < time_f:
            sd_t = sd[t]
            error += np.sum(divide((C_time[t] - M[t]) ** 2, sd_t**2))
            t += 1


    return (error)