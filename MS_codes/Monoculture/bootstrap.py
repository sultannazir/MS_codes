import numpy as np
import csv
import functions as fn
import matplotlib.pyplot as plt

""" INPUT """
path = "Data/nov3_pad3_YFP/monoculture_nov3_pad3_22hr-02_s2t"  # file name until timestamp
fc = "Cy3"  # fluorescent channel name
time_i = 0  # initial time frame
time_f = 23  # final time frame
nump = 50  # number of pixels to combine when resizing
umpp = 0.91  # um per pixel
thresh = 0.1
dthr = 1  # time interval between frames in hr
sd_min = 18 # minimum value of standard deviation in density at every point in space and time
name = "MLE_par/" # name of sub-folder in 'Data' with simulated data at MLE

input = [path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min]

""" STARTING PARAMETER VALUES, SMALLEST STEP SIZE IN PARAMETER SPACE AND BOUNDS"""
k = 1       # conversion factor from Fluorescence intensity to Phase intensity at t=0
l_k, u_k = 0, 100

mu = 0.45/3600        # max growth rate of strain 1 (in 1/sec)
l_mu, u_mu = 0, 2.50 / 3600

Rhf = 150              # cell density parameter to limit growth
l_Rhf, u_Rhf = 0, 1e10

Qhf = 150              # cell density parameter determining dependence of migration on local density
l_Qhf, u_Qhf = 0, 1e10

Phf = 150              # cell density parameter determining dependence of migration on neighboring density
l_Phf, u_Phf = 0, 1e10

Y = 160               # biomass yield (in pixel intensity per unit S0)
l_Y, u_Y = 0, 1e10

Ks = 0.001          # rate of nutrient consumption by Monod kinetics (in unit log10S0)
l_Ks, u_Ks = 0, 1

DC = 1200             # spreading coefficient of cells (in um^2)
l_DC, u_DC = 0, 1e10

DS = 500              # diffusion coefficient of glucose (in um^2/sec)
l_DS, u_DS = 0, 720

l_Parameters = [l_k, l_mu, l_Rhf, l_Qhf, l_Phf, l_Y, l_Ks, l_DC, l_DS]
u_Parameters = [u_k, u_mu, u_Rhf, u_Qhf, u_Phf, u_Y, u_Ks, u_DC, u_DS]


def add_noise(M, sd):
    noise = np.random.normal(0,1,M.shape)*sd
    N = M+noise
    N[N<0] = 0
    return(N)

""" ANNEALING PARAMETERS """
an_iters = 1500
maxTemp = 1
alpha = 0.9
beta = 0.95
max_step = 0

sa_input = [an_iters, maxTemp, alpha, beta, max_step]

""" DEFINE ANNEALING """

def bootstrap_parameter():
    an_iters, maxTemp, alpha, beta, max_step = tuple(sa_input)
    Parameters = np.array([2.5989, 0.00005693, 117.23, 27.46, 20.12, 42.621, 0.0001689, 2656.8, 109.15])

    """ READ DATA """
    Phase, FC = fn.read_data_as_array(input)
    init = FC[0]
    errors = []
    Temps = []
    Parameters_dat = []

    """ GENERATE DATA SET """
    sim_data = []
    t = time_i + 1
    while t < time_f + 1:
        imp = np.loadtxt("Data/" + name + str(t).zfill(2) + ".csv", delimiter=",", dtype=float)
        sim_data.append(imp)
        t += 1
    sim_data = np.array(sim_data)

    sd = fn.get_sd(Phase, FC, input)
    sim_data = add_noise(sim_data, sd)

    # remove noise at cell-free region (excluded data points)
    for i in range(len(sim_data)):
        cut = thresh * np.max(FC[i])
        sim_data[i][FC[i] <= cut] = 0

    current_error = fn.get_error_given_sd(sim_data, sd, init, Parameters, l_Parameters, u_Parameters, input)

    """ BEGIN ITERATIONS """
    iter = 0
    while iter < an_iters:

        Temp = maxTemp * alpha ** iter
        step = 1 + max_step * beta ** iter

        """ make a random move in the Parameter space along log scaled axes """
        new_Parameters = Parameters * 10 ** (np.random.uniform(-0.1, 0.1, len(Parameters)) * step)
        new_error = fn.get_error_given_sd(sim_data, sd, init, new_Parameters, l_Parameters, u_Parameters, input)
        Derror = new_error - current_error

        if Derror <= 0:
            """ Accept all downhill moves """
            Parameters = new_Parameters
            current_error = new_error
            #print('accepted', iter)

        print(current_error)
        errors.append(current_error)
        Temps.append(Temp)
        Parameters_dat.append(Parameters.tolist())

        iter += 1

    print("Error: ", current_error, ", Parameters: ", Parameters)
    data = np.column_stack((Temps, errors, Parameters_dat))
    with open('single_SA_sim_data.txt', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data[-1])

    return errors, Temps, Parameters_dat

bootstrap_parameter()