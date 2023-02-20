import numpy as np
import random
import functions as fn
import os
import copy
from joblib import Parallel, delayed
import csv

idx = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1

""" INPUT """
path = "Data/nov3_pad3_YFP/monoculture_nov3_pad3_22hr-02_s2t"  # file name until timestamp
fc = "Cy3"  # fluorescent channel name
time_i = 0  # initial time frame
time_f = 23  # final time frame
nump = 50  # number of pixels to combine when resizing
umpp = 0.91  # um per pixel
thresh = 0.1  # background subtraction value for phase channel images
dthr = 1  # time interval between frames in hr
sd_min = 21.7 # minimum value of standard deviation in density at every point in space and time

input = [path, fc, time_i, time_f, nump, umpp, thresh, dthr, sd_min]

""" ANNEALING PARAMETERS """
an_iters = 1500
max_wait = 300
maxTemp = 1
alpha = 1

sa_input = [an_iters, max_wait, maxTemp, alpha]


def directed_profile(idx, dirn):
    #sadat = np.loadtxt('profile_par{}.csv'.format(idx), delimiter=',')
    #Parameters = np.delete(sadat, 0, 1)
    Parameters_min = np.array([2.990181,0.000045,267.489776,50.643431,34.425563,51.236493,0.002164,3967.201184,301.910207])

    an_iters, max_wait, maxTemp, alpha = tuple(sa_input)

    """ READ DATA """
    Phase, FC = fn.read_data_as_array(input)

    move = 0
    Parameters = copy.deepcopy(Parameters_min)

    while move <= 1.5:
        move += 0.05 * dirn
        print(move)
        Parameters[idx] = Parameters_min[idx]*10**(move)

        errors = []
        Temps = []
        Parameters_dat = []

        current_error = fn.get_error_NB(Phase, FC, Parameters, input)

        """ BEGIN ITERATIONS """
        iter = 0
        iter_ui = 0
        while (iter < an_iters and iter_ui < max_wait):

            Temp = maxTemp * alpha ** iter

            step_factor = 10**(np.random.uniform(-0.1, 0.1, len(Parameters)))
            step_factor[idx] = 1
            new_Parameters = Parameters * step_factor
            new_error = fn.get_error_NB(Phase, FC, new_Parameters, input)

            Derror = new_error - current_error

            if Derror <= 0:
                """ Accept all downhill moves """
                Parameters = new_Parameters
                current_error = new_error
                iter_ui = 0
                # print('accepted', iter)
            else:
                """ Accept uphill moves with a probability"""
                metropolis = 2.7183 ** (-Derror / Temp)
                rand = random.uniform(0, 1)
                if rand < metropolis:
                    Parameters = new_Parameters
                    current_error = new_error
                    iter_ui = 0
                    # print('accepted', iter)
                else:
                    iter_ui += 1
                # print('declined', iter)

            errors.append(current_error)
            Temps.append(Temp)
            Parameters_dat.append(Parameters.tolist())

            iter += 1

        data = np.column_stack((Temps, errors, Parameters_dat))
        with open('profile_par{}.csv'.format(idx), 'a') as file:
            writer = csv.writer(file)
            writer.writerow(data[-1])


Parallel(n_jobs=2, verbose=9)(delayed(directed_profile)(idx, dirn) for dirn in np.array([-1,1]))



