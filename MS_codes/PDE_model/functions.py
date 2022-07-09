import numpy as np

def parameters_to_global_variables(Parameters):
    ## reads input parameters from dictionary and assigns them as
    ## global variables; variables with names same as the dict keys
    keys = list(Parameters.keys())
    for i in keys:
        globals()[i] = Parameters[i]

def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4 * M
    L += np.roll(M, (0, -1), (0, 1))  # right neighbor
    L += np.roll(M, (0, +1), (0, 1))  # left neighbor
    L += np.roll(M, (-1, 0), (0, 1))  # top neighbor
    L += np.roll(M, (+1, 0), (0, 1))  # bottom neighbor

    return L


def divide(M1, M2):
    """To divide two numpy arrays M1/M2 element-wise and avoide division by zero"""
    f = np.divide(M1, M2, out=np.zeros_like(M1), where=M2 != 0)
    return f


def density_func(M1, M2):
    f = divide(M1, (M1 + M2) * (1 - M1 - M2))
    return f


def den_dep_crossdiffusion(M1, M2):
    """Density dependent cross-diffusion of M1 without constant coefficients"""

    M1_r = np.roll(M1, (0, -1), (0, 1))  # right neighbor
    M1_l = np.roll(M1, (0, +1), (0, 1))  # left neighbor
    M1_t = np.roll(M1, (-1, 0), (0, 1))  # top neighbor
    M1_b = np.roll(M1, (+1, 0), (0, 1))  # bottom neighbor

    M2_r = np.roll(M2, (0, -1), (0, 1))  # right neighbor
    M2_l = np.roll(M2, (0, +1), (0, 1))  # left neighbor
    M2_t = np.roll(M2, (-1, 0), (0, 1))  # top neighbor
    M2_b = np.roll(M2, (+1, 0), (0, 1))  # bottom neighbor

    D = (density_func(M1_r, M2_r) - density_func(M1_l, M2_l)) * (M1_r + M2_r - M1_l - M2_l) / 4 \
        + (density_func(M1_t, M2_t) - density_func(M1_b, M2_b)) * (M1_t + M2_t - M1_b - M2_b) / 4 \
        + density_func(M1, M2) * discrete_laplacian(M1 + M2)

    return D


def update_2d(C1, C2, S, A1, A2):
    dx0 = dx / Len
    dtau = dt * mu1
    res = dtau / dx0 ** 2

    C1diff = res * dC * den_dep_crossdiffusion(C1, C2) + dtau * C1 * S / ((1 + S) * (1 + A2))
    C2diff = res * dC * den_dep_crossdiffusion(C2, C1) + dtau * mu_r * C2 * S / ((1 + S) * (1 + A1))
    Sdiff = res * dS * discrete_laplacian(S) - dtau * beta * (C1 + mu_r * C2) * S / (1 + S)
    A1diff = res * dA1 * discrete_laplacian(A1) - dtau * gamma1 * A1 * C1 / (alpha1 + A1)
    A2diff = res * dA2 * discrete_laplacian(A2) - dtau * gamma2 * A2 * C2 / (alpha2 + A2)

    C1 += C1diff
    C2 += C2diff
    S += Sdiff
    A1 += A1diff
    A2 += A2diff

    # Reflecting boundaries

    C1[0, :], C2[0, :], S[0, :], A1[0, :], A2[0, :] = C1[1, :], C2[1, :], S[1, :], A1[1, :], A2[1, :]  # top
    C1[-1, :], C2[-1, :], S[-1, :], A1[-1, :], A2[-1, :] = C1[-2, :], C2[-2, :], S[-2, :], A1[-2, :], A2[-2,
                                                                                                      :]  # bottom
    C1[:, 0], C2[:, 0], S[:, 0], A1[:, 0], A2[:, 0] = C1[:, 1], C2[:, 1], S[:, 1], A1[:, 1], A2[:, 1]  # left
    C1[:, -1], C2[:, -1], S[:, -1], A1[:, -1], A2[:, -1] = C1[:, -2], C2[:, -2], S[:, -2], A1[:, -2], A2[:, -2]  # right

    return C1, C2, S, A1, A2


def initialize_2D():
    Lx = int(Len/dx)

    C1 = np.zeros((Lx + 1, Lx + 1))
    C2 = np.zeros((Lx + 1, Lx + 1))
    S = np.ones((Lx + 1, Lx + 1)) * S0
    A1 = np.ones((Lx + 1, Lx + 1)) * A10
    A2 = np.ones((Lx + 1, Lx + 1)) * A20

    # sample random points to load bacteria
    n1 = int(np.ceil(f1 * (Lx - 1) * (Lx - 1)))
    n2 = int(np.ceil(f2 * (Lx - 1) * (Lx - 1)))
    ind1 = np.random.randint(low=1, high=int((Lx - 1) * (Lx - 1)), size=n1)
    ind2 = np.random.randint(low=1, high=int((Lx - 1) * (Lx - 1)), size=n2)

    for i in ind1:
        x = int(np.ceil(i / (Lx - 1)))
        y = i % (Lx - 1) + 1
        C1[x, y] = C0

    for i in ind2:
        x = int(np.ceil(i / (Lx - 1)))
        y = i % (Lx - 1) + 1
        C2[x, y] = C0

    return (C1, C2, S, A1, A2)