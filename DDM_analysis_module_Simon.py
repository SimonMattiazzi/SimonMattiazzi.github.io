# -*- coding: utf-8 -*-


"""
Created on Tue Jun  6 13:59:30 2023

c-DDM Analysis module, containing functions for c-DDM analysis.

Measured data for ONE run is packed as a 3D array.
Inside each of 12800 points in k-space there is an array of correlation data (Re and Im).
But there is much more than one run,
there are all possible combinations of magnetic fields, polarizers config, samples ... etc,
stored within a folder tree system.

The idea of this file is to auto-scan through measurement folders and
only select the relevant ones and their relevant kx, ky points,
perform the fits, save them, and then
combine fitted data in many meaningful ways
(in all combinations inside 5 dimensions).

It works for various types of analyses, for examples see "DDM_single_analysis.py" and "DDM_multimeasurement_analysis_2.py"

!!! WARNING !!!: this was once a well-structured module, but it has now become full of pathces and partial and temporary
solutions and it is not as easy to navigate anymore.

Some parts of documentation are not up to date.

Written by Simon

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import datetime
import re
#from numba import njit
#from mpl_toolkits import mplot3d
from tqdm import tqdm
import time
from natsort import natsorted
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# timestamp:
timestamp = str(datetime.datetime.now()).replace(":", ".")[:-10]

# plot parameters:
plt.rcParams.update({"axes.grid" : True,
                     "grid.color": "grey",
                     "grid.linestyle": ":",
                     "figure.dpi": 150,
                     "xtick.minor.visible": True,
                     "ytick.minor.visible": True,
                     "legend.loc": "best",
                     "svg.fonttype": "none"})

# theory constants:
M_all = [1,2,3,4] # E7, GCQ2, N19, N19C
gamma_all = [1,2,3,4] # E7, GCQ2, N19, N19C

### GENERAL FUNCTIONS FOR VARIOUS PURPOSES ###

def set_root(root_ref, canvas1_ref, canvas2_ref):
    """set root for GUI in Fit Corrector"""
    global root
    global canvas1
    global canvas2
    canvas1 = canvas1_ref
    canvas2 = canvas2_ref
    root = root_ref


def filename_update(filename):
    """
    Find a suitable updated filename if the desired name already exists.
    """
    if not os.path.isfile(filename):
        return filename
    else:
        suffix = filename[-4:]
        filename2 = filename[:-4]
        i = 1
        while os.path.isfile(filename):
            filename = filename2 + "_" + str(i) + suffix
            i += 1
        return filename


def save_figure(DESC, overwrite, out_folder=None):
    """
    Save figures to results folder (generated if doesn't exist). If no overwrite is selected, file will get a "_i" suffix

    Parameters:
        desc (str): the description of the figure.\n
        overwrite (bool): Whether to overwrite existing file.
        out_folder: optional, full path

    """
    if out_folder != None:
        out = True
        FOLDER = out_folder
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        filename = os.path.join(FOLDER, f"{DESC}.png")

    else:
        out = False
        if not os.path.exists(os.path.join(FOLDER, "results")): # create results folder in the first run
            os.makedirs(os.path.join(FOLDER,"results"))
        filename = os.path.join(FOLDER, "results", f"{SAMPLE}_{DESC}.png")

    if overwrite == True:
        plt.savefig(filename)
    else:
        if not os.path.isfile(filename):
            plt.savefig(filename)
        else:
            i = 1
            while os.path.isfile(filename):
                if out == True:
                    filename = os.path.join(FOLDER, f"{DESC}_{i}.png")
                    i += 1
                else:
                    filename = os.path.join(FOLDER, "results", f"{SAMPLE}_{DESC}_{i}.png")
                    i += 1
            plt.savefig(filename)


def initial_settings(folder, sample, deltat_):
    """
    Sets SAMPLE and FOLDER globally in the module file.

    Parameters:
        folder: destination FOLDER\n
        sample: SAMPLE name
        deltat: delta t set in DDM experiment settings
        #tau_limit: max 1/tau value in fitting

    """
    global SAMPLE
    global FOLDER
    global deltat
    SAMPLE = sample
    FOLDER = folder
    deltat = deltat_


def initial_settings2(folder, sample):
    """
    Sets SAMPLE and FOLDER globally in the module file.
    Just a quick fix, used where delta_t is not needed.

    Parameters:
        folder: destination FOLDER\n
        sample: SAMPLE name

    """
    global SAMPLE
    global FOLDER
    global deltat
    SAMPLE = sample
    FOLDER = folder


def closest_element_index(lst, target):
    """
   Find the index of the element in the list with the closest value to the target.

   Parameters:
       lst (list or array-like): The list or array of values.
       target (float or int): The target value to find the closest element.

   Returns:
       int: The index of the element in the list that has the closest value to the target.
   """
    arr = np.array(lst)
    closest_index = np.abs(arr - target).argmin()
    return closest_index


def q(k, pixelsize=0.00025/720, pixels=540):
    """
    Calculate the reciprocal space coordinate q from the k-space coordinate k.

    Parameters:
        k (float or array-like): The k-space coordinate(s).
        pixelsize (float, optional): The pixel size. Defaults to 0.00018/512.
        pixels (int, optional): The number of pixels. Defaults to 512.

    Returns:
        float or array-like: The corresponding reciprocal space coordinate(s) q in 1/m
    """
    return k * 2 * np.pi / pixelsize / pixels



def extract_mag_field(run_folder):
    """
    Find the magnetic field value from folder name by searching the appropriate Hall probe results index. Files must be organized in the same tree structure as in the other cases.
    """
    exp_folder = os.path.dirname(os.path.dirname(run_folder))
    match = re.search(r"polarizers/(.*?)/run", run_folder)
    sample = match.group(1)
    match3 = re.search(r"run_(\d+)", run_folder)
    run_orig = match3.group(1)

    xlabel, exp_folder, samplelist, folderlist_full, B_array_full = multifolder_extract(exp_folder, halldata=True)
    for i, samplename in enumerate(samplelist):
        if samplename == sample:
            index_s = i

    for j, folder in enumerate(folderlist_full[index_s]):
        match2 = re.search(r"run_(\d+)", folder)
        run = match2.group(1)
        if run == run_orig:
            index_r = j
            break

    mag_field = B_array_full[index_s][index_r]
    return mag_field


def extract_polarizers_config(run_folder):
    """extract EE or E10O from filepath"""
    if re.search(r"EE", run_folder):
        return "EE"
    elif re.search(r"E10O", run_folder):
        return r"$E_{10}O$"
    else:
        return ""


def npz_to_csv(path, output_folder="D:/IJS report 4/EE polarizers/Results/"):
    """convert any npz file to CSV for further analysis. Subarrays whitin files become new arrays"""

    def flatten_subarrays(data, prefix="", delimiter="_"):
        flat_data = {}
        for key, value in data.items():
            #print (key, value)
            #time.sleep(1)
            if len(value) != 0:
                if isinstance(value, np.ndarray) and isinstance(value[0], np.ndarray):
                    # Recursively flatten subarrays
                    subarray_data = flatten_subarrays(
                        {f"{prefix}{delimiter}{i + 1}": subarray for i, subarray in enumerate(value)}, prefix=key,
                        delimiter=delimiter)
                    flat_data.update(subarray_data)
                else:
                    flat_data[prefix + delimiter + key] = value
        return flat_data

    # Load npz file
    npz_data = np.load(path, allow_pickle=True)
    output_name = os.path.basename(path)[:-4]
    output_path = filename_update(output_folder + output_name + ".csv")

    # Flatten subarrays (including potential sub-sub arrays) using recursion
    flat_data = flatten_subarrays(npz_data)

    # Extract names and lengths
    names = list(flat_data.keys())
    max_length = max(len(value) if isinstance(value, np.ndarray) else 1 for value in flat_data.values())

    # Create a dictionary to store data for each column
    columns = {name: np.full(max_length, np.nan) for name in names}

    # Fill columns with data
    for name, value in flat_data.items():
        if isinstance(value, np.ndarray):
            columns[name][:len(value)] = value
        else:
            columns[name][:] = value

    # Save to CSV using np.savetxt
    header = ",".join(names)
    data = np.column_stack([columns[name] for name in names])
    print(output_path)
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')


def magnetic_field(hall_voltage_V, ratio=1.27):
    """
    Convert the Hall probe output voltage to magnetic field strength.

    Parameters:
    - hall_voltage_V (float): Hall probe output voltage in volts.
    - ratio (float): Ratio between side and middle measurements (default: 1.27).

    Returns:
    - Magnetic field strength in millitesla (mT).
    """
    magnetic_field = (64/2) * (hall_voltage_V - 2.5) / ratio
    return magnetic_field

#############################
### FUNCTIONS FOR FITTING ###
#############################

def fit_corr(corr, t, kx, ky, tolerance=0.0001, init_pars=[100, 1, 0.01, 600, 0.01], bounds=([1, 0, 0, 100, 0], [3000, 1, 1, np.inf, 1]), showplot = False, plotsave=False, overwrite=False, old_return=True, canvas1=False, curr=None, mag_field="", pol_config="", out_folder=None, plotshow=False, cutoff=0.6):
    """
    Fits the correlation function in a given (kx, ky) point.
    If the error is large enough (tolerance), fitting with two exponential functions is used.
    FOR USE WITH DIFFERENT MATERIALS: MANAULLY ADAPT FIT BOUNDS!

    Parameters:
        corr (numpy.ndarray): 2D array representing the correlation data.
        t (numpy.ndarray): 1D array representing the time values.
        kx (int): The k_x value.
        ky (int): The k_y value.
        tolerance (float): The tolerance sigmatau / fittau where 2-exp fit should be used.
        init_pars (list): Initial parameter values for the fit.
        bounds (tuple): Bounds for the fit parameters.
        showplot (bool): Whether to show the plot of the fit (default is False).
        plotsave (bool): Whether to save the plot (default is False).
        overwrite (bool): Whether to overwrite existing file (default is False).
        old_return (bool): Whether to return 4 params (True, default) or 6 (with C2, tau2) + popt and pcov.
        canvas1 (bool): option where we use this function in FitCorrector GUI
        curr (None or int): current to write into saved plot file
        mag_field (str or float): mag field to write onto plot
        pol_config (str): polarizers config to write onto plot
        out_folder (str or None): where to save plot
        plotshow (bool): show plot flag
        cutoff (float): where to cutoff fiting

    Returns:
        tuple: A tuple containing the fitC0, fittau(=1/tau), sigmatau, and sigmaC0 values.
    """
    if showplot == True:
        plotshow = True
    kx_plot = np.fft.fftfreq(len(corr), 1/len(corr))[kx] # sort correctly just for plot label (the whole array is sorted in later steps). Indexing works anyway, because fftfreq [-kx] = - kx.
    twoexp = False


    def fit_func(t, f, C0, y0):  # f = 1 / tau
         return C0 * np.exp(- f * t) + y0


    # 2-exp fit option:
    def fit_func2(t, f, C0, C1, f1, y0):  # f = 1 / tau
        if f < f1 and C1 < C0 and f1 / f > 3: # second must be larger - also adapt bounds and init_pars!
            return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t) + y0
        else:
            return np.inf


    try:
        # try with one exponent:
        popt, pcov = curve_fit(fit_func, t[:int(len(t) * cutoff)], corr[kx, ky][:int(len(t) * cutoff)], p0=[10, 1, 0.01])
        fitC0, fittau, fity0, sigmatau, sigmaC0, sigmay0 = popt[1], popt[0], popt[2] ,np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1]),np.sqrt(pcov[2, 2])
        # if one-exp fit doesnt work, use two-exp:

        if sigmatau / fittau > tolerance:
            twoexp = True
            popt, pcov = curve_fit(fit_func2, t[:int(len(t) * cutoff)], corr[kx, ky][:int(len(t) * cutoff)],
                                   p0=init_pars,
                                   bounds=bounds) # we set initial estimations so that the roles of both exponent terms remain the same.
            fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
            fitC1, fittau2 = popt[2], popt[3] # for initial parameters loop or for corrector
            fity0, sigmay = popt[4], np.sqrt(pcov[4][4])

        if sigmatau / fittau > 1: # tau = 1/tau
            fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan

    except:
        #print("Could not perform fit.")
        fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan

    if plotshow == True or plotsave == True:
        try:
            plt.cla()
            #plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"data, $k_\parallel$ = {kx_plot}, $k_\perp$ = {ky}")
            plt.scatter(t, np.abs(corr[kx, ky]), marker="o", s=15, c="silver", edgecolors="dimgrey",  label=rf"data, $q_\parallel$ = {q(kx_plot):.2e}, $q_\perp$ = {q(ky):.2e}")
            plt.xscale("log")

            if twoexp == True:
                plt.semilogx(t[20:int(len(t) * cutoff)], fit_func2(t[20:int(len(t) * cutoff)], *popt), c="black", linestyle="--",
                             label = "fit:\n" + rf"$C_0$ = {round(popt[1],3)} $\pm$ {round(np.sqrt(pcov[1,1]),3)}"
                             + "\n" +
                             rf"$C_1$ = {round(popt[2], 2)} $\pm$ {round(np.sqrt(pcov[2, 2]), 2)}"
                             + "\n" +
                             rf"$y_0$ = {round(popt[4], 3)} $\pm$ {round(np.sqrt(pcov[4, 4]), 3)}"
                             + "\n" +
                             rf"$1/\tau$ = {round(popt[0], 2)} /s $\pm$ {round(np.sqrt(pcov[0,0]), 2)} /s"
                             + "\n" +
                             r"$1/\tau_2$ = " + f"{round(popt[3], 2)} /s  $\pm$ {round(np.sqrt(pcov[3,3]), 2)} /s")
            else:
                plt.semilogx(t[7:int(len(t) * cutoff)], fit_func(t[7:int(len(t) * cutoff)], *popt), c="black", linestyle="--",
                             label = rf"fit, $C_0$ = {round(popt[1], 3)} $\pm$ {round(np.sqrt(pcov[1,1]),3)}"
                             + "\n" +
                             rf"      $1/\tau$ = {round(popt[0], 2)} /s $\pm$ {round(np.sqrt(pcov[0,0]), 2)} /s"
                             + "\n" +
                             rf"      $y_0$ = {round(popt[2], 2)} /s $\pm$ {round(np.sqrt(pcov[2, 2]), 2)} /s"
                             )

            plt.title("Correlation function")
            try:
                plt.suptitle("c-DDM: " + SAMPLE + ", " + str(round(mag_field,2)) + " mT, " + pol_config)
            except:
                plt.suptitle("c-DDM: " + SAMPLE)
            plt.xlabel("time (s)")
            plt.ylabel("correlation")

            if kx > len(corr) / 2:
                kx_name = - (len(corr) / 2 - 0.5) + kx - (len(corr) / 2 + 0.5)
            else:
                kx_name = kx

            if plotsave == True: #curr, mag_field, out_folder, plotshow
                plt.legend()
                save_figure(f"corr_func_fit_{SAMPLE}_{curr}_mA_{round(mag_field, 1)}_mT_kx{int(kx_name)}_ky{int(ky)}",
                            overwrite=overwrite, out_folder=out_folder)  ###
            if plotshow == True:
                plt.legend()
                if canvas1:
                    print("plotting from fitcorr")
                    canvas2.draw()
                else:
                    plt.show()

        except:
            try: # at least show data
                plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"data, $k_\parallel$ = {kx_plot}, $k_\perp$ = {ky}")
                plt.xlabel("time (s)")
                plt.ylabel("correlation")
                plt.title("Correlation function")
                try:
                    plt.suptitle("c-DDM: " + SAMPLE + ", " + str(round(mag_field, 2)) + " mT, " + pol_config)
                except:
                    plt.suptitle("c-DDM: " + SAMPLE)
                plt.legend()

                if plotsave == True:  # curr, mag_field, out_folder, plotshow
                    plt.legend()
                    save_figure(
                        f"corr_func_fit_{SAMPLE}_{curr}_mA_{round(mag_field, 1)}_mT_kx{int(kx_name)}_ky{int(ky)}",
                        overwrite=overwrite, out_folder=out_folder)  ###
                if plotshow == True:
                    plt.legend()
                    if canvas1:
                        print("plotting from fitcorr")
                        canvas2.draw()
                    else:
                        plt.show()

            except:
                print("Plotting the correlation function was not possible.")

    if old_return==True:
        return fitC0, fittau, sigmatau, sigmaC0 #original

    else:
        if twoexp:
            return fitC0, fittau, sigmatau, sigmaC0, fitC1, fittau2, popt, pcov #original
        else:
            return fitC0, fittau, sigmatau, sigmaC0, popt, pcov  # original


def plot_fit_from_existing(corr, t, kx, ky, popt, pcov, plotshow, plotsave=False, overwrite=False, out_folder=None, canvas10=False, deriv=False, curr="", mag_field="", pol_config="", cutoff=0.7):
    """
       Plots the correlation function fit using existing (saved) fit parameters.
       Either from full fit (2D arrays for all kx/ky's) or temp fit corrector file (just for one point)

       Parameters:
           corr (numpy.ndarray): 2D array representing the correlation data.
           t (numpy.ndarray): 1D array representing the time values.
           kx (int): The k_x value.
           ky (int): The k_y value.
           popt (numpy.ndarray): Fitted parameters obtained from a previous fit.
           pcov (numpy.ndarray): Covariance matrix of the fitted parameters.
           plotshow (bool): Whether to show the plot (default is False).
           plotsave (bool): Whether to save the plot (default is False).
           overwrite (bool): Whether to overwrite existing file (default is False).
           out_folder (str or None): Output folder for saving the plot.
           canvas10 (bool): option in GUI FitCorrector
           deriv (bool): wheter to plot 2nd order derivative roots as vlines for easier comparison in FitCorrector
           curr (str): current value for saving filename
           mag_field (str or float): magfield value for saving filename
           pol_config (str): polarizers config to print on plot
           cutoff (float): where to stop fitting
       """

    def fit_func(t, f, C0, y0):  # f = 1 / tau
        return C0 * np.exp(- f * t) + y0


    # 2-exp fit option:
    def fit_func2(t, f, C0, C1, f1, y0):  # f = 1 / tau
        if f < f1 and C1 < C0 and f1 / f > 3:  # second must be larger - also adapt bounds and init_pars!
            return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t) + y0
        else:
            return np.inf


    if canvas10:
        plt.clf()
        #fig=plt.figure(dpi=100)

    plt.cla()
    if kx > len(corr) / 2:
        kx_name = - (len(corr)/2 - 0.5) + kx - (len(corr) / 2 + 0.5)
    else:
        kx_name = kx

    plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"data, $k_\parallel$ = {int(kx_name)}, $k_\perp$ = {ky}")

    try:
        if len(popt) < 4:
            twoexp = False
        else:
            twoexp = True

        if twoexp == True:
            plt.semilogx(t[:int(len(t) * cutoff)], fit_func2(t[:int(len(t) * cutoff)], *popt), c="black", linestyle="--",
                         label="fit:\n" + rf"$C_0$ = {round(popt[1], 3)} $\pm$ {round(np.sqrt(pcov[1, 1]), 3)}"
                               + "\n" +
                               rf"$C_1$ = {round(popt[2], 2)} $\pm$ {round(np.sqrt(pcov[2, 2]), 2)}"
                               + "\n" +
                               rf"$y_0$ = {round(popt[4], 3)} $\pm$ {round(np.sqrt(pcov[4, 4]), 3)}"
                               + "\n" +
                               rf"$1/\tau$ = {round(popt[0], 2)} /s $\pm$ {round(np.sqrt(pcov[0, 0]), 2)} /s"
                               + "\n" +
                               r"$1/\tau_2$ = " + f"{round(popt[3], 2)} /s  $\pm$ {round(np.sqrt(pcov[3, 3]), 2)} /s")
        else:
            plt.semilogx(t[:int(len(t) * cutoff)], fit_func(t[:int(len(t) * cutoff)], *popt), c="black", linestyle="--",
                         label=rf"fit, $C_0$ = {round(popt[1], 3)} $\pm$ {round(np.sqrt(pcov[1, 1]), 3)}"
                               + "\n" +
                               rf"      $1/\tau$ = {round(popt[0], 2)} /s $\pm$ {round(np.sqrt(pcov[0, 0]), 2)} /s"
                               + "\n" +
                               rf"      $y_0$ = {round(popt[2], 2)} /s $\pm$ {round(np.sqrt(pcov[2, 2]), 2)} /s"
                         )

        plt.title("Correlation function")
        try:
            plt.suptitle("c-DDM: " + SAMPLE + ", " + str(round(mag_field, 2)) + " mT, " + pol_config)
        except:
            plt.suptitle("c-DDM: " + SAMPLE)

        plt.xlabel("time (s)")
        plt.ylabel("correlation")

        # plot roots of 2nd order derivative, where taus should be:
        if deriv:

            def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
                if f > f1:  # first must be larger!
                    return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
                else:
                    return np.inf

            try:
                twoexp = True
                popt, pcov = curve_fit(fit_func2, t[7:], corr[kx, ky][7:],
                                       p0=[300, 1, 1, 1],
                                       bounds=([1, 0, 0, 0], [3000, 10, 10, 200]))
                fitC0, fittau, sigmatau, sigmaC0, fittau2 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1]), \
                                                            popt[3]
                success = True

                if sigmatau / fittau > 1:  # unreliable fit
                    fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
                    success = False

            except:
                fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
                success = False

            if success:
                # find the roots of 2nd order derivative. They should be very close to tau 1 and tau 2.
                y = fit_func2(t, *popt)
                grad2 = np.gradient(np.gradient(np.abs(y), np.log10(t)), np.log10(t))
                sign = np.sign(grad2)
                dif = np.diff(sign)
                crossings_plus = np.where((dif == 2) | (dif == 1))[0]

            if len(crossings_plus) == 2:
                p1, p2 = [np.log10(t)[crossings_plus[0]], grad2[crossings_plus[0]]],  [np.log10(t)[crossings_plus[1]], grad2[crossings_plus[1]]]
                plt.axvline(x=t[crossings_plus[0]],
                            label=f"2nd order derivative roots: \n1/t = {round(1 / t[crossings_plus[0]], 1)} 1/s", c="C3",
                            linestyle="-.")
                plt.axvline(x=t[crossings_plus[1]], label=f"1/t = {round(1 / t[crossings_plus[1]], 1)} 1/s", c="C3",
                            linestyle=":")

            if len(crossings_plus) == 1:
                p1 = [np.log10(t)[crossings_plus[0]], grad2[crossings_plus[0]]]
                plt.axvline(x=t[crossings_plus[0]],
                            label=f"2nd order derivative roots: \n 1/t = {round(1 / t[crossings_plus[0]], 1)} 1/s", c="C3",
                            linestyle="-.")
    except:
        plt.title("Correlation function")


    if plotsave == True:
        plt.legend()
        try:
            save_figure(f"corr_func_fit_{SAMPLE}_{curr}_mA_{round(mag_field,1)}_mT_kx{int(kx_name)}_ky{int(ky)}", overwrite=overwrite, out_folder=out_folder) ###
        except:
            save_figure(f"corr_func_fit_{SAMPLE}_000_mA_000_mT_kx{int(kx_name)}_ky{int(ky)}",
                        overwrite=overwrite, out_folder=out_folder)
    if plotshow == True:
        plt.legend()
        if canvas10:
            print("plotting fit from ex")
            canvas1.draw()
        else:
            plt.show()
    else:
        plt.cla()

#########################################
### FUNCTIONS FOR SINGLE RUN ANALYSIS ###
#########################################

def plot_corr(corr, t, kx, ky, plotsave=False, overwrite=False):
    """
    Plots the correlation function for a given (kx, ky) point.

    Parameters:
        corr: Correlation functions data
        kx (int): The k_x value.\n
        ky (int): The k_y value.\n
        plotsave (bool): Whether to save the plot. \n
        overwrite (bool): Whether to overwrite existing file.

    """
    kx_plot = np.fft.fftfreq(len(corr), 1/len(corr))[kx] # sort correctly just for plot label (the whole array is sorted in later steps). Indexing works anyway, because fftfreq [-kx] = - kx.

    plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"$k_\parallel$={kx_plot}, $k_\perp$={ky}")
    plt.title(rf"Correlation function, $k_\parallel$={kx_plot}, $k_\perp$={ky}")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("time (s)")
    plt.ylabel("correlation")
    #plt.legend()

    if plotsave == True:
        save_figure(f"corr_func_{kx_plot}_{ky}", overwrite=overwrite)

    plt.show()


def plot_2D(data, cmap="plasma", levels=30, plotsave=False, overwrite=False):
    """
    Plots the 2D graph of the fitted 1/tau values in every (kx, ky) point.

    Parameters:
        data (numpy.ndarray): Fitted 1/tau values for the whole grid to plot.\n
        levels (int): The number of contour levels (default is 30).\n
        cmap: Cmap for the plot. Default is "plasma"\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """
    # have to be plotted with 2 contributions (left/ right), otherwise there is a connecting "roof"
    # have to rearange the data from [0, ... , 63, -63, -62, ... 1]
    # to [-63, ...., 0, ... 63]

    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index + 1]
    data_plus = data[half_index + 1:]

    data = np.concatenate((data_plus, data_minus))
    x = np.arange(min(x), max(x) + 1)

    X, Y = np.meshgrid(q(x), q(y))
    plt.contour(X, Y, np.transpose(data), linewidths=0.2,colors="black", levels=levels)
    plt.contourf(X, Y, np.transpose(data), cmap=cmap, levels=20)

    plt.title(r"Fitted correlation times 1 / $\tau$")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("$q_\parallel (1/m)$")
    plt.ylabel("$q_\perp$ (1/m)")
    plt.colorbar(label=r"1 / $\tau$ (1/s)")

    if plotsave == True:
        save_figure("tau_2D", overwrite=overwrite)

    plt.show()


def plot_3D(data, cmap="plasma", plotsave=False, overwrite=False, z_lim=None, alpha=0.9):
    """
    Plots the 3D surface of the fitted 1/tau data in every (kx, ky) point.
    There is also a part that takes care of NaN values, so 3D plot surface can be shown.

    Parameters:
        data (numpy.ndarray): Fitted 1/tau values for the whole grid to plot.\n
        cmap: Cmap for the plot. Default is "plasma"\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """

    # have to be plotted with 2 contributions (left/ right), otherwise there is a connecting "roof"
    # have to rearange the data from [0, ... , 63, -63, -62, ... 1]
    # to [-63, ...., 0, ... 63]
    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))

    # take care for Nan values - replace with closest neighbour that is not Nan:
    nanarray = np.argwhere(np.isnan(data) * 1 == 1) # convert bool to int with * 1
    print(f"Start: {len(np.argwhere(np.isnan(data) * 1 == 1))} Nan values")
    for pair in nanarray:
        xa, ya = pair[0], pair[1]
        # find the closest working value:
        i_updt = 1
        while True:
            if ya + i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya + i_updt]), axis=1)):
                data[xa, ya] = data[xa, ya + i_updt]
                break
            if ya - i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya - i_updt]), axis=1)):
                data[xa, ya] = data[xa, ya - i_updt]
                break
            if xa + i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa + i_updt, ya]), axis=1)):
                data[xa, ya] = data[xa + i_updt, ya]
                break
            if xa - i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa - i_updt, ya]), axis=1)):
                data[xa, ya] = data[xa - i_updt, ya]
                break
            i_updt += 1

    print(f"End: {len(np.argwhere(np.isnan(data) * 1 == 1))} Nan values")

    x = np.arange(min(x), max(x) + 1)

    plt.figure(figsize=(6, 6))

    X, Y = np.meshgrid(q(x), q(y))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, np.transpose(data), cmap=cmap, alpha=alpha)

    ax.set_zlim(0, z_lim)
    ax.set_title(r"c-DDM: " + SAMPLE + "\n" + r"Fitted correlation times $1/\tau$" )
    ax.set_xlabel("$q_\parallel (1/m)$")
    ax.set_ylabel("$q_\perp (1/m)$")
    ax.set_zlabel(r"1/$\tau (1/s)$")
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.xaxis._axinfo["grid"]['linestyle'] = "-"
    ax.yaxis._axinfo["grid"]['linestyle'] = "-"
    ax.zaxis._axinfo["grid"]['linestyle'] = "-"

    if plotsave == True:
        save_figure("tau_3D", overwrite=overwrite)

    plt.show()


def plot_1D(data, ky, plotsave=False, overwrite=False):
    """
    Plots a 1D graph of the calculated 1/tau values for a given k_y-value slice.
    Currently x axis is squared!!

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        ky (number): The k_y value where we want to slice.\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """

    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]

    data = np.concatenate((data_plus, data_minus))
    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(y == ky)[0][0]

    # plot:
    # change q**2 to q!!!
    plt.plot(q(x)**2, np.transpose(data)[j])

    plt.title(rf"Fitted correlation times 1 / $\tau$, $k_y=${ky} slice")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("$q_\parallel^2 (1/m^2)$")
    plt.ylabel(r"$1/\tau$ (1/s)")

    if plotsave == True:
        save_figure(f"tau_slice_ky_{ky}", overwrite=overwrite)

    plt.show()


def plot_1D_y(data, kx, plotsave=False, overwrite=False):
    """
    Plots a 1D graph of the calculated 1/tau values for a given k_x-value slice.
    Currently x axis is squared!!!

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        kx (number): The k_x value where we want to slice.\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """

    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))

    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(x == kx)[0][0]

    # plot:
    plt.plot(q(y)**2, data[j])
    plt.title(rf"Fitted correlation times 1 / $\tau$, $k_x=${kx} slice")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("$q_\perp^2 (1/m^2)$")
    plt.ylabel(r"$1/\tau$ (1/s)")

    if plotsave == True:
        save_figure(f"tau_slice_kx_{kx}", overwrite=overwrite)
    plt.show()



def fit_k_tau(data, sigmatau, ky, add_suptitle="", plotshow=True, plotsave=True, overwrite=False, output_folder="D/IJS Report 4", sample="undef"):
    #temp!!! see squared (_sq) below
    """
    Fits the 1D graph of the calculated 1/tau values for a given k_y-value slice.
    Model:
    y = a(x-c)**2 + b(x-c)

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        ky (number): The k_y value where we want to fit.\n
        plotshow (bool): Whether to show the plot of the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).

    Returns:
        tuple: A tuple containing the fit a, fit c, fit b, sigma a, sigma c and sigma b values.
    """


    # def fit_func2(x, a, b, d):
    #     return a * (x-0)**2 + b * (x-0) + d


    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))

    Ddata_minus = sigmatau[:half_index +1 ]
    Ddata_plus = sigmatau[half_index + 1:]
    Ddata = np.concatenate((Ddata_plus, Ddata_minus))

    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(y == ky)[0][0]

    # fit:
    # Remove NaN values
    mask = ~np.isnan(np.transpose(data)[j])
    x_valid = x[mask]
    y_valid = np.transpose(data)[j][mask]

    #popt, pcov = curve_fit(fit_func2, q(x_valid), y_valid)

    # plot:
    if plotshow == True:
        plt.errorbar(q(x), np.transpose(data)[j], yerr=np.transpose(Ddata)[j], label="data")
        #plt.plot(q(x)**2, fit_func2(q(x), *popt), c="black", linestyle="--",
                         # label = r"quadratic fit:"+"\n"+rf"$x_0$ = {round(popt[1],8)} $\pm$ {round(np.sqrt(pcov[1,1]),6)}"
                         # + "\n" +
                         # rf"$a$ = {round(popt[0], 13)}$\pm$ {round(np.sqrt(pcov[0,0]), 15)}"
                         # + "\n" +
                         # rf"$b$ = {round(popt[1], 7)}$\pm$ {round(np.sqrt(pcov[1,1]), 7)}"
                         #  + "\n" +
                         # rf"$d$ = {round(popt[2], 2)}$\pm$ {round(np.sqrt(pcov[2,2]), 3)}")
        plt.title(rf"Fitted correlation times 1 / $\tau$, $q_\perp=${q(ky)} slice")
        plt.suptitle("c-DDM: " + SAMPLE + ", " + add_suptitle)
        plt.xlabel("$q_\parallel (1/m)$")
        plt.ylabel(r"$1/\tau (1/s)$")
        plt.legend()

        if plotsave == True:
            save_figure(f"quadr_fit_tau_slice_{ky}", overwrite=overwrite)

        plt.show()
        
        #temporary!!
        name=filename_update(output_folder+f"/Results/slice_ky{ky}_{sample}.npz")
        np.savez(filename_update(name),
                 q_para_arr=q(x),
                 oneovertau_arr=np.transpose(data)[j],
                 sigmatau_arr=np.transpose(Ddata)[j])
        npz_to_csv(name, output_folder=output_folder+"/Results/")

    #return popt[0], popt[1], popt[2], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])



def fit_k_tau_y(data, sigmatau, kx, add_suptitle="", plotshow=True, plotsave=True, overwrite=False, output_folder="D/IJS Report 4", sample="undef"):
    #temp!!! see squared (_sq) below
    """
    Fits the 1D graph of the calculated 1/tau values for a given k_y-value slice.
    Model: y = a(x-c)**2 + b(x-c)

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        ky (number): The k_y value where we want to fit.\n
        plotshow (bool): Whether to show the plot of the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).

    Returns:
        tuple: A tuple containing the fit a, fit c, fit b, sigma a, sigma c and sigma b values.
    """


    # def fit_func2(x, a, b, d):
    #     return a * (x-0)**2 + b * (x-0) + d


    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))
    Ddata_minus = sigmatau[:half_index +1 ]
    Ddata_plus = sigmatau[half_index + 1:]
    Ddata = np.concatenate((Ddata_plus, Ddata_minus))

    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(x == kx)[0][0]

    # plot:
    plt.plot(q(y)**2, data[j])
    plt.title(rf"Fitted correlation times 1 / $\tau$, $k_x=${kx} slice")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("$q_\perp^2 (1/m^2)$")
    plt.ylabel(r"$1/\tau$ (1/s)")

    # fit:
    # popt, pcov = curve_fit(fit_func2, q(y)[3:], data[j][3:])

    # plot:
    if plotshow == True:
        plt.clf()
        plt.errorbar(q(y)**1, data[j], yerr=Ddata[j], label="data")
        # plt.plot(q(y)**2, fit_func2(q(y), *popt), c="black", linestyle="--",
        #                  label = r"quadratic fit:"+"\n"+rf"$x_0$ = {round(popt[1],8)} $\pm$ {round(np.sqrt(pcov[1,1]),6)}"
        #                  + "\n" +
        #                  rf"$a$ = {round(popt[0], 13)}$\pm$ {round(np.sqrt(pcov[0,0]), 15)}"
        #                  + "\n" +
        #                  rf"$b$ = {round(popt[1], 7)}$\pm$ {round(np.sqrt(pcov[1,1]), 7)}"
        #                   + "\n" +
        #                  rf"$d$ = {round(popt[2], 2)}$\pm$ {round(np.sqrt(pcov[2,2]), 3)}")
        plt.title(rf"Fitted correlation times 1 / $\tau$, $q_\parallel=${q(kx)} slice")
        plt.suptitle("c-DDM: " + SAMPLE + ", " + add_suptitle)
        plt.xlabel(r"$q_\perp (1/m)$")
        plt.ylabel(r"$1/\tau (1/s)$")
        plt.legend()

        if plotsave == True:
            save_figure(f"quadr_fit_tau_slice_{kx}x", overwrite=overwrite)

        plt.show()
        
        #temporary!!
        # name=filename_update(f"D:/Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/Results/slice_kx{kx}.npz")
        # np.savez(filename_update(f"D:/Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/Results/slice_kx{kx}.npz"),
        #          q_perp_arr=q(y),
        #          oneovertau_arr=data[j])
        # npz_to_csv(name, output_folder="D:/Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/Results/")

        name=filename_update(output_folder+f"/Results/slice_kx{kx}_{sample}.npz")
        np.savez(filename_update(name),
                 q_perp_arr=q(y),
                 oneovertau_arr=data[j],
                 sigmatau_arr = Ddata[j])
        npz_to_csv(name, output_folder=output_folder+"/Results/")
        

    # return popt[0], popt[1], popt[2], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])




def fit_k_tau_sq(data, ky, add_suptitle="", plotshow=True, plotsave=True, overwrite=False):
    """
    Fits the 1D graph of the calculated 1/tau values for a given k_y-value slice.
    Model:
    y = a(x-c)**2 + b(x-c)

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        ky (number): The k_y value where we want to fit.\n
        plotshow (bool): Whether to show the plot of the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).

    Returns:
        tuple: A tuple containing the fit a, fit c, fit b, sigma a, sigma c and sigma b values.
    """


    def fit_func2(x, a, b, d):
        return a * (x-0)**2 + b * (x-0) + d


    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]

    data = np.concatenate((data_plus, data_minus))
    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(y == ky)[0][0]

    # fit:
    # Remove NaN values
    mask = ~np.isnan(np.transpose(data)[j])
    x_valid = x[mask]
    y_valid = np.transpose(data)[j][mask]

    popt, pcov = curve_fit(fit_func2, q(x_valid), y_valid)

    # plot:
    if plotshow == True:
        plt.plot(q(x)**2, np.transpose(data)[j], label="data")
        plt.plot(q(x)**2, fit_func2(q(x), *popt), c="black", linestyle="--",
                         label = r"quadratic fit:"+"\n"+rf"$x_0$ = {round(popt[1],8)} $\pm$ {round(np.sqrt(pcov[1,1]),6)}"
                         + "\n" +
                         rf"$a$ = {round(popt[0], 13)}$\pm$ {round(np.sqrt(pcov[0,0]), 15)}"
                         + "\n" +
                         rf"$b$ = {round(popt[1], 7)}$\pm$ {round(np.sqrt(pcov[1,1]), 7)}"
                          + "\n" +
                         rf"$d$ = {round(popt[2], 2)}$\pm$ {round(np.sqrt(pcov[2,2]), 3)}")
        plt.title(rf"Fitted correlation times 1 / $\tau$, $k_y=${ky} slice")
        plt.suptitle("c-DDM: " + SAMPLE + ", " + add_suptitle)
        plt.xlabel("$q_\parallel^2 (1/m^2)$")
        plt.ylabel(r"$1/\tau (1/s)$")
        plt.legend()

        if plotsave == True:
            save_figure(f"quadr_fit_tau_slice_{ky}", overwrite=overwrite)

        plt.show()

    return popt[0], popt[1], popt[2], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])


def fit_k_tau_y_sq(data, kx, add_suptitle="", plotshow=True, plotsave=True, overwrite=False):
    """
    Fits the 1D graph of the calculated 1/tau values for a given k_y-value slice.
    Model: y = a(x-c)**2 + b(x-c)

    Parameters:
        data (numpy.ndarray): The calculated 1/tau values.\n
        ky (number): The k_y value where we want to fit.\n
        plotshow (bool): Whether to show the plot of the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).

    Returns:
        tuple: A tuple containing the fit a, fit c, fit b, sigma a, sigma c and sigma b values.
    """


    def fit_func2(x, a, b, d):
        return a * (x-0)**2 + b * (x-0) + d


    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]

    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))

    x = np.arange(min(x), max(x) + 1)

    # calculate which index corresponds to the given k_y:
    j = np.argwhere(x == kx)[0][0]

    # plot:
    plt.plot(q(y)**2, data[j])
    plt.title(rf"Fitted correlation times 1 / $\tau$, $k_x=${kx} slice")
    plt.suptitle("c-DDM: " + SAMPLE)
    plt.xlabel("$q_\perp^2 (1/m^2)$")
    plt.ylabel(r"$1/\tau$ (1/s)")

    # fit:
    popt, pcov = curve_fit(fit_func2, q(y)[3:], data[j][3:])

    # plot:
    if plotshow == True:
        plt.clf()
        plt.plot(q(y)**2, data[j], label="data")
        plt.plot(q(y)**2, fit_func2(q(y), *popt), c="black", linestyle="--",
                         label = r"quadratic fit:"+"\n"+rf"$x_0$ = {round(popt[1],8)} $\pm$ {round(np.sqrt(pcov[1,1]),6)}"
                         + "\n" +
                         rf"$a$ = {round(popt[0], 13)}$\pm$ {round(np.sqrt(pcov[0,0]), 15)}"
                         + "\n" +
                         rf"$b$ = {round(popt[1], 7)}$\pm$ {round(np.sqrt(pcov[1,1]), 7)}"
                          + "\n" +
                         rf"$d$ = {round(popt[2], 2)}$\pm$ {round(np.sqrt(pcov[2,2]), 3)}")
        plt.title(rf"Fitted correlation times 1 / $\tau$, $k_x=${kx} slice")
        plt.suptitle("c-DDM: " + SAMPLE + ", " + add_suptitle)
        plt.xlabel("$q_parallel^2 (1/m^2)$")
        plt.ylabel(r"$1/\tau (1/s)$")
        plt.legend()

        if plotsave == True:
            save_figure(f"quadr_fit_tau_slice_{kx}x", overwrite=overwrite)

        plt.show()

    return popt[0], popt[1], popt[2], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])


def plot_3D_fitted_parabolas(data, ky_array, fit_a_array, fit_c_array, fit_b_array, fit_d_array, plotsurface=True, plotlines=True, plotfit=True, plotsave=False, overwrite=False):
    """
    Plots the 3D graph of the data with fitted parabolas for all ky's.
    There is also a part that takes care of NaN values, so it can be plotted as 3D surface.
    Parameters:
        data (numpy.ndarray): The fitted 1/tau values for the whole (kx, ky) grid to plot.\n
        fit_a_array (numpy.ndarray): The array of fitted a values.\n
        fit_c_array (numpy.ndarray): The array of fitted c values.\n
        plotsurface (bool): Whether to plot the surface (default is True).\n
        plotlines (bool): Whether to plot the slice lines (default is True).\n
        plotfit (bool): Whether to plot the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """
    x = np.fft.fftfreq(len(data), 1/len(data))
    half_index = int(len(x)/2)
    y = x[: half_index + 1]
    data_minus = data[:half_index +1 ]
    data_plus = data[half_index + 1:]
    data = np.concatenate((data_plus, data_minus))

    # take care of NaN values
    nanarray = np.argwhere(np.isnan(data) * 1 == 1) # convert bool to int with * 1
    for pair in nanarray:
        xa, ya = pair[0], pair[1]
        # find the closest working value:
        i_updt = 1
        while True:
            if ya + i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya + i_updt]), axis=1)):
                data[xa, ya] = data[xa, ya + i_updt]
                break
            if ya - i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya - i_updt]), axis=1)):
                data[xa, ya] = data[xa, ya - i_updt]
                break
            if xa + i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa + i_updt, ya]), axis=1)):
                data[xa, ya] = data[xa + i_updt, ya]
                break
            if xa - i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa - i_updt, ya]), axis=1)):
                data[xa, ya] = data[xa - i_updt, ya]
                break
            i_updt += 1

    x = np.arange(min(x), max(x) + 1)

    plt.figure(figsize=(6, 6))

    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect = (1,2,1))

    if plotsurface == True:
        X, Y = np.meshgrid(q(x), q(y))
        # take care for Nan values:
        nanarray = np.argwhere(np.isnan(data) * 1 == 1) # convert bool to int with * 1
        for pair in nanarray:
            xa, ya = pair[0], pair[1]
            data[xa, ya] = data[xa, ya + 1]
        ax.plot_surface(X, Y, np.transpose(data), cmap="plasma", alpha=0.38)


    ax.set_title(r"c-DDM: " + SAMPLE + "\n" + r"Fitted correlation times $1/\tau$"
                 + "\nand all fitted quadratic functions")
    ax.set_xlabel("$k_\parallel$")
    ax.set_ylabel("$k_\perp$")
    ax.set_zlabel(r"1/$\tau$")
    ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.zaxis._axinfo["grid"]['linewidth'] = 0.1
    ax.xaxis._axinfo["grid"]['linestyle'] = "-"
    ax.yaxis._axinfo["grid"]['linestyle'] = "-"
    ax.zaxis._axinfo["grid"]['linestyle'] = "-"


    def fit_func2(x, a, c, b, d):
        return a * (x-c)**2 + b * (x-c) + d


    for ky in ky_array:
        j = np.argwhere(y == ky)[0][0]
        if plotlines == True:
            ax.plot(q(x), np.transpose(data)[j], zs=q(ky), zdir="y", c="C0", linewidth=0.5)
        if plotfit == True:
            popt = [fit_a_array[j], fit_c_array[j], fit_b_array[j], fit_d_array[j]]
            ax.plot(q(x), fit_func2(q(x), *popt), c="black", linestyle="--", zs=q(ky), zdir="y", linewidth=0.5)

    if plotsave == True:
        save_figure("quadr_fit_all_3D", overwrite=overwrite)

    plt.show()


def linear_fit_a(fit_a_array, ky_array, plotshow=True, plotsave=False, overwrite=False):
    """
    Fits the linear function to the array of fitted $a$ values from quadratic functions.

    Parameters:
        fit_a_array (numpy.ndarray): The array of fit $a$ values.\n
        ky_array (numpy.ndarray): The array of k_y values.\n
        plotshow (bool): Whether to show the plot of the fit (default is True).\n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).

    Returns:
        tuple: A tuple containing the fit k, fit y0, sigma k, and sigma y0 values.
    """


    def fit_func_3(x, k, y0):
        return k * x + y0


    popt, pcov = curve_fit(fit_func_3, ky_array, fit_a_array)

    if plotshow == True:
        plt.plot(ky_array, fit_a_array, label = "data")
        plt.title(r"Fitted $a$ parameters")
        plt.suptitle("c-DDM: " + SAMPLE)
        plt.xlabel("$k_\perp$")
        plt.ylabel(r"$a$")
        plt.plot(ky_array, fit_func_3(ky_array, *popt), c="black", linestyle="--",
                                 label = "linear fit:"+"\n"+rf"$y_0$ = {round(popt[1],7)} $\pm$ {round(np.sqrt(pcov[1,1]), 7)}"
                                 + "\n" +
                                 rf"$k$ = {round(popt[0], 10)}$\pm$ {round(np.sqrt(pcov[0,0]), 9)}")

        plt.legend()

        if plotsave == True:
            save_figure("lin_fit_a", overwrite=overwrite)

        plt.show()

    return popt[0], popt[1], np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])


def amplitude_imshow(var1, var2, corr, vmax=None, cmap="plasma", plotsave=False, overwrite=False):
    """
    Plots the amplitude of the signal.

    Parameters:
        var1 (numpy.ndarray): Values from 1st camera \n
        var2 (numpy.ndarray): Values from 2nd camera \n
        corr: Correlation functions data \n
        vmax: max value (used to cut off weird values)\n
        cmap: colormap used for imshow \n
        plotsave (bool): Whether to save the plot (default is False). \n
        overwrite (bool): Whether to overwrite existing file (default is False).
    """
    # amplitude = (average(signal1, signal2))**1/4 * corr at time 0:
    amplitude = np.abs(corr[...,0]) * ((var1 + var2) / 2)**0.25

    # subplots settings:
    plt.title("c-DDM amplitude: " + SAMPLE)
    plt.xlabel(r"$k_\perp$")
    plt.ylabel(r"$k_\parallel$")
    plt.imshow(amplitude[1:-1], cmap=cmap, vmax=vmax)
    plt.colorbar()

    if plotsave == True:
        save_figure("amplitude_2D" + SAMPLE, overwrite=overwrite)

    plt.show()

###########################################################
### FUNCTIONS FOR ANALYZING AND COMPARING MULTIPLE RUNS ###
###########################################################

def multifolder_extract(exp_folder, suffix="", description="", halldata=False):
    '''
    Perform a scan through the folders with results from experiments with different magnetic fields,
    extract relevant information and create arrays, used in later analysis:
        - array of sample names
        - 2D array of different magnetic field values folder (array of 1D arrays)
        - list of 1D arrays of magnetic field values at each experiment

    We may use Hall probe results file to determine magnetic field values.
    If Hall file is not present, el. current values will be used for x-axis.

    The data must be stored in the following folder structure:

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...

    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
    - exp_folder (str): Path to the folder containing all samples' experiments.\n
    - suffix (str, optional): Suffix of the Hall probe data filename. Defaults to "".\n
    - description (str, optional): Description of the saved plot filename. Defaults to "".\n
    - halldata (bool, optional): Flag to indicate whether to use Hall probe data for B values. Defaults to False.\n

    Returns:
    - xlabel (str): Label for the x-axis in subsequent plots (depends on the Hall data).
    - exp_folder (str): Path to the experiment folder containing all samples' experiments.
    - samplelist_full (list): List of sample names within the experiment folder.
    - folderlist_full (list): List of lists containing the folders within each sample.
    - B_array_full (list): List of arrays containing the B values for each folder.
    '''

    samplelist = os.listdir(exp_folder)

    samplelist_full = []
    folderlist_full = []
    B_array_full = []
    for FOLDER in samplelist:
        if os.path.isdir(exp_folder + "/" + FOLDER) == True and FOLDER != "Results": # skip files, keep folders

            samplelist_full.append(FOLDER)
            sample = os.path.basename(FOLDER)
            initial_settings2(exp_folder, sample)
            FOLDER = exp_folder + "/" + FOLDER
            B_array = np.array([]) # for current or magnetic field values
            xlabel = r"I [mA] (1 A $\approx$ 30 mT)"

            # find all folders and make B array from them:
            # natsorted.
            folderlist = natsorted(os.listdir(FOLDER))

            for i in range(len(folderlist)):
                if os.path.isdir(FOLDER + "/" + folderlist[i]) == True and folderlist[i] != "results": # skip files, keep folders
                    # create an array with B values from each experiment:
                    try: # decimal number exctraction:
                        B_array = np.append(B_array, float(re.findall("(?<!run_)|(?<!run_\d)-?\d+\.\d+", folderlist[i])[0]))
                                                                        #(?<!run_)-?\d+\.\d+ (orig. decimal)
                                                                        # (?<!run_)-?\d+ (orig non-decimal)
                    except: # non-decimal number exctraction:
                        B_array = np.append(B_array, float(re.findall("(?<!run_)(?<!run_\d)-?\d+", folderlist[i])[0]))
                    folderlist[i] = FOLDER + '/' + str(folderlist[i]) # make absolute path
                else: folderlist[i] = None
            folderlist = [x for x in folderlist if x is not None] # keep only folders
            folderlist_full.append(folderlist)

            if halldata == True: # Update B array with B values, calculated from Hall probe
                current, volt_hall = np.loadtxt(FOLDER+"/results/DDM_results_Hall_" + suffix + ".txt", delimiter=",", unpack=True)

                if not isinstance(volt_hall, (list, np.ndarray)):  # if only one value
                    volt_hall = [volt_hall]

                if len(volt_hall) == len(B_array): # if something is wrong, just use current values
                    xlabel = "B [mT]"
                    B_array = np.array([])
                    for volt in volt_hall:
                        B_array = np.append(B_array, magnetic_field(volt/1000))
                else: print("Legnth mismatch. Using current values instead.")
            B_array_full.append(B_array)

    return xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full


def multimeasurement_comparison_B(exp_folder, kx, ky, deltat, suffix="", description="", add_suptitle="", tolerance=0.5, halldata=False, show_fit_plots=False, save_fit_plots=False, showplot=True, plotsave=False, overwrite=False, use_existing_fit=True, mode=1, theory=False):
    '''
    Perform a comparison of measurements across different magnetic field values.
    We choose certain kx, ky values. Then we calculate 1/tau in in this point and compare it over different B values (different folders).

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.

    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
    - exp_folder (str): Path to the folder containing all samples' experiments.\n
    - kx (float): Value of kx.\n
    - ky (float): Value of ky.\n
    - suffix (str, optional): Suffix of the Hall probe data filename. Defaults to "".\n
    - description (str, optional): Description of the saved plot filename. Defaults to "".\n
    - add_title (str, optional): add some text to the suptitle (e.g. plarizers configuration) \n
    - tolerance (float, optional): The tolerance sigmatau / fittau where 2-exp fit should be used. Defaults to 0.5 \n
    - halldata (bool, optional): Flag to indicate whether to use Hall probe data for B values. Defaults to False.\n
    - show_fit_plot (bool, optional): Flag to indicate whether to show the individual fit plots. Defaults to True.\n
    - showplot (bool, optional): Flag to indicate whether to show the final plot. Defaults to True.\n
    - plotsave (bool, optional): Flag to indicate whether to save the plot. Defaults to False.\n
    - overwrite (bool, optional): Flag to indicate whether to overwrite existing saved plot. Defaults to False.\n
    - use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n

    Returns:
    - final_oneovertau_array (list): List of arrays containing the values of 1/tau for each magnetic field value.
    '''

    final_B_array, final_oneovertau_array, final_sigmatau_array, final_sample = [], [], [], []
    xlabel, exp_folder, samplelist, folderlist_full, B_array_full = multifolder_extract(exp_folder, suffix=suffix, description="test1", halldata=halldata)#, halldata=False, show_fit_plots=False, showplot=True, plotsave=PLOTSAVE, overwrite=OVERWRITE)
    #tqdm(range(len(corr)), desc="Fitting tau values", ncols=100, colour="#82e0aa")
    for i, sample in enumerate(samplelist):

        initial_settings2(exp_folder, sample)
        folderlist = folderlist_full[i]
        B_array = B_array_full[i]
        oneovertau_array = np.array([])
        sigmatau_array = np.array([])
        print(f"sample: {sample}")

        colourbase = "#8bd89d"
        percent_ini = 50
        percent_ini_copy = percent_ini
        change = False
        while np.min([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ]) < 17:
            percent_ini += 1
            change = True
        if change == True and i == 0:
            print(f"Changed initial color of loading bar from {percent_ini_copy} percent to {percent_ini} percent.")
        highest = np.max([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ])
        delta = (255 * percent_ini / highest - percent_ini) / len(samplelist)
        colour = "#" + hex(int(int(colourbase[1:3], 16) * (percent_ini + delta * 0.9 * i) / 100))[2:] + hex(int(int(colourbase[3:5], 16) * (percent_ini + delta * 0.9 * i) / 100))[2:]+ hex(int(int(colourbase[5:7], 16) * (percent_ini + delta * 1 * i) / 100))[2:]


        mag_ind = - 1
        for SUBFOLDER in tqdm(folderlist, ncols=100, colour=colour):
            mag_ind += 1
            mag_field = B_array[mag_ind]

            curr_f = folderlist[mag_ind]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
            pattern = r"current_(.*?)_mA"
            match = re.search(pattern, curr_f)
            if match:
                curr = match.group(1)
            else:
                print("No match found in the folder string.")

            # analysis:
            if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz") == True) and use_existing_fit == True:
                #print("using old")
                try:
                    loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                    fit_C0_array, fit_tau_array, sigma_C0_array, sigma_tau_array = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                    fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array[kx, ky], fit_tau_array[kx, ky], sigma_tau_array[kx, ky], sigma_C0_array[kx, ky]
                except:
                    print("")
                    datapc = np.load(
                        SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                        allow_pickle=True)
                    popt, pcov = datapc["popt"], datapc["pcov"]
                    # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                    fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]

                if show_fit_plots == True or save_fit_plots == True:
                    try:
                        data = np.load(SUBFOLDER + "/" +  "corr.npz")
                        t, corr = data["t"] * deltat, data["corr"]
                        try:
                            datapc = np.load(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz", allow_pickle=True)
                            popt, pcov = datapc["popt"], datapc["pcov"]
                            print("using old but correct")
                        except:
                            datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                            popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]
                        plot_fit_from_existing(corr, t, kx, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_B_kx{kx}_ky{ky}")
                    except:
                        "Exception showing fit plot"

            else:
                data = np.load(SUBFOLDER + "/" +  "corr.npz")
                t, corr = data["t"] * deltat, data["corr"]
                fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kx, ky=ky, tolerance=tolerance, showplot=show_fit_plots, plotshow=show_fit_plots, mag_field=mag_field, curr=curr, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_B_kx{kx}_ky{ky}")

            oneovertau_array = np.append(oneovertau_array, fittau)
            sigmatau_array = np.append(sigmatau_array, sigmatau)

        if len(folderlist) == 0:
            print("Empty folders problem!")
        # update final arrays:
        final_B_array.append(B_array)
        final_oneovertau_array.append(oneovertau_array)
        final_sigmatau_array.append(sigmatau_array)
        final_sample.append(sample)

    # plot final arrays:
    plt.clf()
    expBarr=[]
    exptauarr=[]
    expsigarr=[]
    #-------------
    tau_theor_arr = []
    #-------------
    for i in range(len(final_B_array)):
        plt.errorbar(final_B_array[i], final_oneovertau_array[i], yerr=final_sigmatau_array[i], c=f"C{i}", capsize=4, label=final_sample[i])
        plt.scatter(final_B_array[i], final_oneovertau_array[i], c=f"C{i}")

        #----------------------------------------------------------------#
        if theory:
            sample=samplelist[i]
            # theoretical values
            # extract right mode and M and gamma
            if extract_polarizers_config(exp_folder) == "EE":
                mode = 1
            elif extract_polarizers_config(exp_folder) == r"$E_{10}O$":
                mode = 2
            else: print("Cannot extract polarizers config - mode unknown!")

            if sample == "E7":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "GCQ2":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "N19":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "N19C":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            else: print("Cannot extract sample name - M, gamma_skl unknown!")

            #plot:
            plt.plot(final_B_array[i], tau_theor(qperp=q(ky), qpara=q(kx), B=np.array(final_B_array[i]), M=M, gamma_skl=gamma_skl, mode=mode), label=f"theoretical {sample}")
            # export:
            tau_theor_arr.append(np.array(tau_theor(qperp=q(ky), qpara=q(kx), B=np.array(final_B_array[i]), M=M, gamma_skl=gamma_skl, mode=mode)))
            #print(tau_theor_arr)
        #----------------------------------------------------------------#

    # export data:
    #DTYPE OBJECT!!!
    name=filename_update(exp_folder+f"/Results/multi_compare_B_kx{kx}_ky{ky}.npz")
    np.savez(filename_update(exp_folder+f"/Results/multi_compare_B_kx{kx}_ky{ky}.npz"),
             #samples=np.array(final_sample,dtype="object"),
             final_B_array=np.array(final_B_array,dtype="object"),
             final_oneovertau_array=np.array(final_oneovertau_array,dtype="object"),
             final_sigmatau_array=np.array(final_sigmatau_array,dtype="object"),
             tau_theor_arr = np.array(tau_theor_arr,dtype="object")) ######
    npz_to_csv(name, output_folder=exp_folder+"/Results/")

    plt.xlabel(xlabel)
    plt.ylabel(r"1/$\tau$[1/s]")
    plt.suptitle(r"$\tau^{-1}(B)$ dependence" + " - " + add_suptitle)
    #plt.suptitle(r"$\tau^{-1}(B)$ dependence - uncrossed polarizers")
    plt.title(f"($k_\parallel$ = {kx}, $k_\perp$ = {ky}) point")
    plt.legend()

    if plotsave == True:
        initial_settings(exp_folder, description)
        save_figure("multimeasurement", overwrite=overwrite)

    if showplot == True:
        plt.show()

def multimeasurement_comparison_ky_slice(FOLDER, ky_sl, B_target, deltat, tolerance=0.2, show_fit_plots=False, save_fit_plots=False, halldata=True, add_suptitle=r"$EE$ polarizers", use_existing_fit = True):
    """
    Perform multi-measurement comparison for a given slice of k_y.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.

    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
        FOLDER (str): The main folder path containing the data.
        ky_sl (int): The k_y index of the desired k-space slice.
        B_target (float or int): The target magnetic field value.
        tolerance (float, optional): The tolerance for fitting. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to show plots for fitting. Defaults to False.
        halldata (bool, optional): Whether the data is hall data or not. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"$EE$ polarizers".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n
    """

    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)

    for i, sample in enumerate(samplelist_full):
        initial_settings2(exp_folder, sample)
        folderlist = folderlist_full[i]
        B_array = B_array_full[i]
        B_ind = closest_element_index(B_array, B_target)
        mag_field = B_array[B_ind]

        curr_f = folderlist[B_ind]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
        pattern = r"current_(.*?)_mA"
        match = re.search(pattern, curr_f)
        if match:
            curr = match.group(1)
        else:
            print("No match found in the folder string.")
        #print(curr)

        for ind, SUBFOLDER in enumerate(folderlist):
            if ind == B_ind:

                data = np.load(SUBFOLDER + "/" +  "corr.npz")
                t, corr = data["t"] * deltat, data["corr"]

                # LOOP ONLY THROUGH THE WHOLE DESIRED SLICE OF K-SPACE AND FIT TAU IN EVERY POINT:
                fit_tau_array, fit_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))
                sigma_tau_array, sigma_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))

                print("\n")
                # Iterate over the correlation function's indices for fitting tau values
                for kx in range(len(corr)):
                    for ky in range(len(corr[0])):
                        if ky == ky_sl:
                            try:
                                if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz") == True)and use_existing_fit == True:

                                    try:
                                        loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                        fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                        fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky]
                                    except:
                                        print("")
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                        fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]

                                    if show_fit_plots == True or save_fit_plots == True:
                                        #data = np.load(SUBFOLDER + "/" +  "corr.npz") # needed because of corr length, takes 6 microseconds
                                        #t, corr = data["t"] * deltat, data["corr"]
                                        try:
                                            try:
                                                datapc = np.load(
                                                    SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                                                    allow_pickle=True)
                                                popt, pcov = datapc["popt"], datapc["pcov"]
                                                print("using old but correct")
                                            except:
                                                datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                                popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]

                                            plot_fit_from_existing(corr, t, kx, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_ky{ky}_slice_B{B_target}")
                                        except:
                                            print("Exception showing fit plot.")

                                else:
                                    fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kx, ky=ky, tolerance=tolerance, showplot=show_fit_plots, mag_field=mag_field,curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_ky{ky}_slice_B{B_target}")

                                # Store the fitted values in the respective arrays
                                fit_C0_array[kx, ky] = fitC0
                                fit_tau_array[kx, ky] = fittau
                                sigma_C0_array[kx, ky] = sigmaC0
                                sigma_tau_array[kx, ky] = sigmatau
                            except:
                                print("Exception - fitting error")

                # PLOT 2D PLOT OF A GIVEN SLICE (ky = const.) OF ALL TAU VALUES:
                data = fit_tau_array
                x = np.fft.fftfreq(len(data), 1/len(data))
                half_index = int(len(x)/2)
                y = x[: half_index + 1]

                data_minus = data[:half_index +1 ]
                data_plus = data[half_index + 1:]

                data = np.concatenate((data_plus, data_minus))
                x = q(np.arange(min(x), max(x) + 1))

                # calculate which index corresponds to the given k_y:
                j = np.argwhere(y == ky_sl)[0][0]

                # plot:
                plt.plot(x, np.transpose(data)[j], label=sample)

                if halldata == True:
                    unit = " mT"
                    quant = "B = "
                else:
                    unit = " mA"
                    quant = "I = "

                plt.title(rf"Fitted correlation times 1 / $\tau$, $k_y=${ky_sl} slice, "  + quant + str(round(B_array[ind], 1)) + unit)
                plt.suptitle("c-DDM: " + add_suptitle)
                plt.xlabel("$q_\parallel$ $(1/m)$")
                plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend()
    plt.show()


def multimeasurement_comparison_kx_slice(FOLDER, kx_sl, B_target, deltat, tolerance=0.2, show_fit_plots=False, save_fit_plots = False, halldata=True, add_suptitle=r"$EE$ polarizers", use_existing_fit = True):
    """
    Perform multi-measurement comparison for a given slice of k_x.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.

    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
        FOLDER (str): The main folder path containing the data.
        kx_sl (int): The k_x index of the desired k-space slice.
        B_target (float or int): The target magnetic field value.
        tolerance (float, optional): The tolerance for fitting. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to show plots for fitting. Defaults to False.
        halldata (bool, optional): Whether the data is hall data or not. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"$EE$ polarizers".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n
    """
    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)

    for i, sample in enumerate(samplelist_full):
        initial_settings2(exp_folder, sample)
        folderlist = folderlist_full[i]
        B_array = B_array_full[i]
        B_ind = closest_element_index(B_array, B_target)
        mag_field = B_array[B_ind]

        curr_f = folderlist[B_ind]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
        pattern = r"current_(.*?)_mA"
        match = re.search(pattern, curr_f)
        if match:
            curr = match.group(1)
        else:
            print("No match found in the folder string.")

        for ind, SUBFOLDER in enumerate(folderlist):
            if ind == B_ind:

                # IMPORT DATA:
                data = np.load(SUBFOLDER + "/" +  "corr.npz")
                t, corr = data["t"] * deltat, data["corr"]

                # LOOP ONLY THROUGH THE WHOLE DESIRED SLICE OF K-SPACE AND FIT TAU IN EVERY POINT:
                fit_tau_array, fit_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))
                sigma_tau_array, sigma_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))

                print("\n")
                # Iterate over the correlation function's indices for fitting tau values

                kx = kx_sl
                for ky in range(len(corr[0])):
                    try:
                        if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz")==True) and use_existing_fit == True:
                            try:
                                loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky]
                            except:
                                print("")
                                datapc = np.load(
                                    SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                                    allow_pickle=True)
                                popt, pcov = datapc["popt"], datapc["pcov"]
                                # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]

                            if show_fit_plots == True or save_fit_plots == True:
                                #data = np.load(SUBFOLDER + "/" +  "corr.npz") # needed because of corr length, takes 6 microseconds
                                #t, corr = data["t"] * deltat, data["corr"]
                                try:
                                    try:
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        print("using old but correct")
                                    except:

                                        datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                        popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]


                                    plot_fit_from_existing(corr, t, kx, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_kx{kx}_slice_B{B_target}")
                                except:
                                    print("Exception showing fit plots.")

                        else:
                            #data = np.load(SUBFOLDER + "/" +  "corr.npz")
                            #t, corr = data["t"] * deltat, data["corr"]
                            fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kx, ky=ky, tolerance=tolerance, showplot=show_fit_plots, plotshow=show_fit_plots, curr=curr, mag_field=mag_field, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_kx{kx}_slice_B{B_target}")

                        # Store the fitted values in the respective arrays
                        fit_C0_array[kx, ky] = fitC0
                        fit_tau_array[kx, ky] = fittau
                        sigma_C0_array[kx, ky] = sigmaC0
                        sigma_tau_array[kx, ky] = sigmatau
                    except:
                        print("Exception - fitting error")

                data = fit_tau_array
                x = np.fft.fftfreq(len(data), 1/len(data))
                half_index = int(len(x)/2)
                y = x[: half_index + 1]

                data_minus = data[:half_index + 1]
                data_plus = data[half_index + 1:]
                data = np.concatenate((data_plus, data_minus))

                x = np.arange(min(x), max(x) + 1)
                # calculate which index corresponds to the given k_x:
                j = np.argwhere(x == kx_sl)[0][0]
                y = q(y)
                # plot:
                if halldata == True:
                    unit = " mT"
                    quant = "B = "
                else:
                    unit = " mA"
                    quant = "I = "

                plt.plot(y, data[j], label=sample)
                plt.title(rf"Fitted correlation times 1 / $\tau$, $k_x=${kx_sl} slice, "  + quant + str(round(B_array[ind], 1)) + unit)
                plt.suptitle("c-DDM: " + add_suptitle)
                plt.xlabel("$q_\perp$ $(1/m)$")
                plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend()
    plt.show()


def multimeasurement_comparison_different_qs_y(FOLDER, samplename, deltat, ky_arr=[0, 1, 3], kx=0, tolerance=0.2, show_fit_plots=False, save_fit_plots = False, halldata=True, add_suptitle=r"", use_existing_fit = True, theory=False):
    """
    Perform multi-measurement comparison for one sample at different B values for different q vectors along the y-direction.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
    The data must be stored in the following folder structure (current value folde names must contain numbers!):

    The data must be stored in the following folder structure:

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz, etc.)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz, etc.)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...

    Parameters:
        FOLDER (str): The path to the main folder containing the data.
        samplename (str): The name of the sample to analyze.
        ky_arr (list, optional): A list of y-component q values to analyze. Defaults to [0, 1, 3].
        kx (float, optional): The x-component q value to be used for analysis. Defaults to 0.
        tolerance (float, optional): The tolerance for fitting the correlation function. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to display plots for fitting the correlation function. Defaults to False.
        halldata (bool, optional): Whether the data contains Hall probe results for determining magnetic field values.
            If False, el. current values will be used to extract B values. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n

    Returns:
        None
    """


    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)

    str_ky = str(ky_arr)
    # check if the desired samplename even exists:
    samp_ex = False
    for samp in samplelist_full:
        if samp == samplename:
            samp_ex = True

    if samp_ex != True:
        print("SAMPLENAME DOES NOT EXIST!")

    for i, sample in enumerate(samplelist_full):
        if sample == samplename: # only look at one sample
            initial_settings2(exp_folder, sample)
            folderlist = folderlist_full[i]
            B_array = B_array_full[i]
            final_multiarray = np.zeros((len(ky_arr), len(B_array))) # used to store final values
            final_multiarray_sig = np.zeros((len(ky_arr), len(B_array)))  # used to store final values
            final_multiarray_ampl = np.zeros((len(ky_arr), len(B_array))) # used to store final values

            # for each q, we scan through all fields:
            for ind1, kyi in enumerate(ky_arr):
                print(rf"$k_\perp$={kyi}")

                colourbase = "#37847b"
                percent_ini = 50
                percent_ini_copy = percent_ini
                change = False
                while np.min([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ]) < 17:
                    percent_ini += 1
                    change = True
                if change == True and i == 0:
                    print(f"Changed initial color of loading bar from {percent_ini_copy} percent to {percent_ini} percent.")
                highest = np.max([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ])
                delta = (255 * percent_ini / highest - percent_ini) / len(ky_arr)
                colour = "#" + hex(int(int(colourbase[1:3], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:] + hex(int(int(colourbase[3:5], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:]+ hex(int(int(colourbase[5:7], 16) * (percent_ini + delta * 1 * ind1) / 100))[2:]


                for ind2, SUBFOLDER in enumerate(tqdm(folderlist, ncols=100, colour=colour)): # for each B
                    # IMPORT DATA:
                    data = np.load(SUBFOLDER + "/" +  "corr.npz")
                    t, corr = data["t"] * deltat, data["corr"]

                    datavar = np.load(SUBFOLDER + "/" +  "var.npz")
                    var1, var2 = datavar["var1"], datavar["var2"]

                    amplitude = np.abs(corr[...,0]) #* ((var1 + var2) / 2)**0.25
                    final_multiarray_ampl[ind1][ind2] = amplitude[kx][kyi]

                    mag_field = B_array[ind2]

                    curr_f = folderlist[ind2]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                    pattern = r"current_(.*?)_mA"
                    match = re.search(pattern, curr_f)
                    if match:
                        curr = match.group(1)
                    else:
                        print("No match found in the folder string.")
                    # only check the desired kx, ky point:
                    try:
                        if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz")==True) and use_existing_fit == True:
                            try:
                                loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, kyi], fit_tau_array1[kx, kyi], sigma_tau_array1[kx, kyi], sigma_C0_array1[kx, kyi]
                            except:
                                print("")
                                datapc = np.load(
                                    SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz",
                                    allow_pickle=True)
                                popt, pcov = datapc["popt"], datapc["pcov"]
                                # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]

                            if show_fit_plots == True or save_fit_plots == True:
                                try:
                                    try:
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        print("using old but correct")
                                    except:

                                        datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                        popt, pcov = datapc["popt_2D"][int(kx), int(kyi)], datapc["pcov_2D"][int(kx), int(kyi)]
                                    plot_fit_from_existing(corr, t, kx, kyi, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_different_qs_kx_{kx}_ky_{str_ky}_{samplename}")
                                except:
                                    print("Exception showing fit plots.")
                                    print(popt)

                        else:
                            fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kx, ky=kyi, tolerance=tolerance, showplot=show_fit_plots, plotshow=show_fit_plots, curr=curr, mag_field=mag_field, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_different_qs_kx_{kx}_ky_{str_ky}_{samplename}")

                        final_multiarray[ind1][ind2] = fittau
                        final_multiarray_sig[ind1][ind2] = sigmatau
                    except:
                        print("Exception - fitting error")

    # PLOT 2D PLOT:
    #-------------
    tau_theor_arr = [] # it would be more consistent if tau_theor_arr would be created simultaneoously with final_multiarray etc
    #-------------
    for ind3, kyi in enumerate(ky_arr):
        #plt.plot(B_array, final_multiarray[ind3], label=str(kyi))
        plt.scatter(B_array, final_multiarray[ind3], c=f"C{ind3}")
        plt.errorbar(B_array, final_multiarray[ind3], yerr=final_multiarray_sig[ind3], c=f"C{ind3}", capsize=4)
        plt.plot(B_array, final_multiarray[ind3],  c=f"C{ind3}", label=str(round(q(kyi), -3)))
        #----------------------------------------------------------------#
        if theory:
            sample=samplename
            # theoretical values
            # extract right mode and M and gamma
            if extract_polarizers_config(exp_folder) == "EE":
                mode = 1
            elif extract_polarizers_config(exp_folder) == r"$E_{10}O$":
                mode = 2
            else: print("Cannot extract polarizers config - mode unknown!")

            if sample=="E7":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample=="GCQ2":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample=="N19":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample=="N19C":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            else: print("Cannot extract sample name - M, gamma_skl unknown!")

            #plot:
            plt.plot(B_array, tau_theor(qperp=q(kyi), qpara=q(kx), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode), label=f"theoretical {sample}")
            # export:
            tau_theor_arr.append(np.array(tau_theor(qperp=q(kyi), qpara=q(kx), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode)))
            print(tau_theor_arr)
        #----------------------------------------------------------------#

    # extract data:
    name=filename_update(exp_folder + f"/Results/multi_compare_different_qs_kx_{kx}_ky_{str_ky}_{samplename}.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_compare_different_qs_kx_{kx}_ky_{str_ky}_{samplename}.npz"),
             B_array=B_array,
             final_multiarray=final_multiarray,
             final_multiarray_sig=final_multiarray_sig,
             tau_theor_arr = tau_theor_arr)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")

    plt.title(rf"Fitted correlation times 1 / $\tau$ for different $q_\perp$'s, $q_\parallel$ = {q(kx)}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$\mu_0 H$ $(mT)$")
    plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend(title=r"$q_\perp$ (1/m)")
    plt.show()


def multimeasurement_comparison_different_qs_x (FOLDER, samplename, deltat, kx_arr=[0, 1, 3], ky=0, tolerance=0.2, show_fit_plots=False, save_fit_plots = False, halldata=True, add_suptitle=r"", use_existing_fit=True, theory=False):
    """
    Perform multi-measurement comparison for one sample at different B values for different q vectors.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
        FOLDER (str): The main folder path containing the data.

        tolerance (float, optional): The tolerance for fitting. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to show plots for fitting. Defaults to False.
        halldata (bool, optional): Whether the data is hall data or not. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n
    """

    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)
    str_kx = str(kx_arr)

    for i, sample in enumerate(samplelist_full):
        if sample == samplename: # only look at one sample
            initial_settings2(exp_folder, sample)
            folderlist = folderlist_full[i]
            B_array = B_array_full[i]
            final_multiarray = np.zeros((len(kx_arr), len(B_array))) # used to store final values
            final_multiarray_sig = np.zeros((len(kx_arr), len(B_array)))  # used to store final values
            final_multiarray_ampl = np.zeros((len(kx_arr), len(B_array))) # used to store final values

            # for each q, we scan through all fields:
            for ind1, kxi in enumerate(kx_arr):
                colourbase = "#996ef1"
                percent_ini = 40
                percent_ini_copy = percent_ini
                change = False
                while np.min([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ]) < 17:
                    percent_ini += 1
                    change = True
                if change == True and i == 0:
                    print(f"Changed initial color of loading bar from {percent_ini_copy} percent to {percent_ini} percent.")
                highest = np.max([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ])
                delta = (255 * percent_ini / highest - percent_ini) / len(kx_arr)
                colour = "#" + hex(int(int(colourbase[1:3], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:] + hex(int(int(colourbase[3:5], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:]+ hex(int(int(colourbase[5:7], 16) * (percent_ini + delta * 1 * ind1) / 100))[2:]

                for ind2, SUBFOLDER in enumerate(tqdm(folderlist, ncols=100, colour=colour)): # for each B
                    mag_field = B_array[ind2]
                    curr_f = folderlist[ind2]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                    pattern = r"current_(.*?)_mA"
                    match = re.search(pattern, curr_f)
                    if match:
                        curr = match.group(1)
                    else:
                        print("No match found in the folder string.")

                    # IMPORT DATA:
                    data = np.load(SUBFOLDER + "/" +  "corr.npz")
                    t, corr = data["t"] * deltat, data["corr"]

                    datavar = np.load(SUBFOLDER + "/" +  "var.npz")
                    var1, var2 = datavar["var1"], datavar["var2"]

                    amplitude = np.abs(corr[...,0]) #* ((var1 + var2) / 2)**0.25
                    final_multiarray_ampl[ind1][ind2] = amplitude[kxi][ky]

                    # only check the desired kx, ky point:
                    try:
                        if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz")==True) and use_existing_fit == True:
                            try:
                                loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kxi, ky], fit_tau_array1[kxi, ky], sigma_tau_array1[kxi, ky], sigma_C0_array1[kxi, ky]
                            except:
                                print("")
                                datapc = np.load(
                                    SUBFOLDER + f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz",
                                    allow_pickle=True)
                                popt, pcov = datapc["popt"], datapc["pcov"]
                                #fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0],pcov[1][1]

                                #ERROR: LOOP DOESNT COVER SCENARIO WHEN YOU WANT TO USE CORRECETED VALUES FROM "OLD BUT CORRECTED" IF YOU DONT PLOT!
                            if show_fit_plots == True or save_fit_plots == True:
                                #data = np.load(SUBFOLDER + "/" +  "corr.npz") # needed because of corr length, takes 6 microseconds
                                #t, corr = data["t"] * deltat, data["corr"]
                                try:
                                    try:
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        print("using old but corrected")
                                    except:
                                        datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                        popt, pcov = datapc["popt_2D"][int(kxi), int(ky)], datapc["pcov_2D"][int(kxi), int(ky)]
                                    plot_fit_from_existing(corr, t, kxi, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_different_qs_ky_{ky}_kx_{str_kx}_{samplename}")
                                except:
                                    print("Exception showing fit plots.")

                        else:
                            #print("again")
                            #data = np.load(SUBFOLDER + "/" +  "corr.npz")
                            #t, corr = data["t"] * deltat, data["corr"]
                            fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kxi, ky=ky, tolerance=tolerance, curr=curr, mag_field=mag_field, showplot=show_fit_plots, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_compare_different_qs_ky_{ky}_kx_{str_kx}_{samplename}")

                        final_multiarray[ind1][ind2] = fittau
                        final_multiarray_sig[ind1][ind2] = sigmatau
                    except:
                        print("Exception - fitting error")

    # PLOT 2D PLOT:
    #-------------
    tau_theor_arr = [] # it would be more consistent if tau_theor_arr would be created simultaneoously with final_multiarray etc
    #-------------
    for ind3, kxi in enumerate(kx_arr):
        plt.scatter(B_array, final_multiarray[ind3], c=f"C{ind3}")
        plt.errorbar(B_array, final_multiarray[ind3], yerr=final_multiarray_sig[ind3], c=f"C{ind3}", capsize=4)
        plt.plot(B_array, final_multiarray[ind3],  c=f"C{ind3}", label=str(round(q(kxi), -3)))
 #----------------------------------------------------------------#
        if theory:
            sample=samplename
            # theoretical values
            # extract right mode and M and gamma
            if extract_polarizers_config(exp_folder) == "EE":
                mode = 1
            elif extract_polarizers_config(exp_folder) == r"$E_{10}O$":
                mode = 2
            else: print("Cannot extract polarizers config - mode unknown!")

            if sample == "E7":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "GCQ2":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "N19":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            elif sample == "N19C":
                M = M_all[0]
                gamma_skl = gamma_all[0]
            else: print("Cannot extract sample name - M, gamma_skl unknown!")
            #plot:
            plt.plot(B_array, tau_theor(qperp=q(ky), qpara=q(kxi), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode), label=f"theoretical {sample}")
            # export:
            tau_theor_arr.append(np.array(tau_theor(qperp=q(ky), qpara=q(kxi), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode)))
        #----------------------------------------------------------------#

    # extract data:
    name=filename_update(exp_folder + f"/Results/multi_compare_different_qs_ky_{ky}_kx_{str_kx}_{samplename}.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_compare_different_qs_ky_{ky}_kx_{str_kx}_{samplename}.npz"),
             B_array=B_array,
             final_multiarray=final_multiarray,
             final_multiarray_sig=final_multiarray_sig,
             tau_theor_arr = tau_theor_arr)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")

    plt.title(rf"Fitted correlation times 1 / $\tau$ for different $q_\parallel$'s, $k_\perp$ = {ky}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$B$ $(mT)$")
    plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend(title=r"$q_\parallel$ (1/m)", loc="upper right")
    plt.show()


def multimeasurement_comparison_different_qs_x_fit (FOLDER, samplename, deltat, kx_arr=[0, 1, 3], ky=0, tolerance=0.2, show_fit_plots=False, save_fit_plots = False, halldata=True, add_suptitle=r"", use_existing_fit = True, theory=False):
    """
    Perform multi-measurement comparison for one sample at different B values for different q vectors.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
        FOLDER (str): The main folder path containing the data.

        tolerance (float, optional): The tolerance for fitting. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to show plots for fitting. Defaults to False.
        halldata (bool, optional): Whether the data is hall data or not. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n
    """

    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)
    str_kx = str(kx_arr)

    for i, sample in enumerate(samplelist_full):
        if sample == samplename: # only look at one sample
            initial_settings2(exp_folder, sample)
            folderlist = folderlist_full[i]
            B_array = B_array_full[i]
            final_multiarray = np.zeros((len(kx_arr), len(B_array))) # used to store final values
            final_multiarray_sig = np.zeros((len(kx_arr), len(B_array)))  # used to store final values
            final_multiarray_ampl = np.zeros((len(kx_arr), len(B_array))) # used to store final values

            # for each q, we scan through all fields:
            for ind1, kxi in enumerate(kx_arr):

                colourbase = "#996ef1"
                percent_ini = 40
                percent_ini_copy = percent_ini
                change = False
                while np.min([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ]) < 17:
                    percent_ini += 1
                    change = True
                if change == True and i == 0:
                    print(f"Changed initial color of loading bar from {percent_ini_copy} percent to {percent_ini} percent.")
                highest = np.max([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ])
                delta = (255 * percent_ini / highest - percent_ini) / len(kx_arr)
                colour = "#" + hex(int(int(colourbase[1:3], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:] + hex(int(int(colourbase[3:5], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:]+ hex(int(int(colourbase[5:7], 16) * (percent_ini + delta * 1 * ind1) / 100))[2:]

                for ind2, SUBFOLDER in enumerate(tqdm(folderlist, ncols=100, colour=colour)): # for each B
                    mag_field = B_array[ind2]
                    curr_f = folderlist[ind2]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                    pattern = r"current_(.*?)_mA"
                    match = re.search(pattern, curr_f)
                    if match:
                        curr = match.group(1)
                    else:
                        print("No match found in the folder string.")

                    # IMPORT DATA:
                    data = np.load(SUBFOLDER + "/" +  "corr.npz")
                    t, corr = data["t"] * deltat, data["corr"]

                    datavar = np.load(SUBFOLDER + "/" +  "var.npz")
                    var1, var2 = datavar["var1"], datavar["var2"]

                    amplitude = np.abs(corr[...,0]) #* ((var1 + var2) / 2)**0.25
                    final_multiarray_ampl[ind1][ind2] = amplitude[kxi][ky]

                    # only check the desired kx, ky point:
                    try:
                        if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz")==True) and use_existing_fit == True:
                            try:
                                loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kxi, ky], fit_tau_array1[kxi, ky], sigma_tau_array1[kxi, ky], sigma_C0_array1[kxi, ky]
                            except:
                                print("")
                                datapc = np.load(
                                    SUBFOLDER + f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz",
                                    allow_pickle=True)
                                popt, pcov = datapc["popt"], datapc["pcov"]
                                # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]
                            if show_fit_plots == True or save_fit_plots == True:
                                #data = np.load(SUBFOLDER + "/" +  "corr.npz") # needed because of corr length, takes 6 microseconds
                                #t, corr = data["t"] * deltat, data["corr"]
                                try:
                                    try:
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kxi}_ky{ky}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        print("using old but correct")
                                    except:
                                        datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                        popt, pcov = datapc["popt_2D"][int(kxi), int(ky)], datapc["pcov_2D"][int(kxi), int(ky)]
                                    plot_fit_from_existing(corr, t, kxi, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}")
                                except:
                                    print("Exception showing fit plots.")

                        else:
                            #data = np.load(SUBFOLDER + "/" +  "corr.npz")
                            #t, corr = data["t"] * deltat, data["corr"]
                            fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kxi, ky=ky, curr=curr, mag_field=mag_field, tolerance=tolerance, showplot=show_fit_plots, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}")

                        final_multiarray[ind1][ind2] = fittau
                        final_multiarray_sig[ind1][ind2] = sigmatau
                    except:
                        print("Exception - fitting error")

    # fit for each k:
    koef_array, offset_array = np.array([]), np.array([])
    er_koef_array, er_offset_array = np.array([]), np.array([])


    def linfit(x, k, a):
        return x * k + a

    # PLOT 2D PLOT:
    fitxarr=[]
    fityarr=[]
    #
    tau_theor_arr = [] # would be more consistent if created at the same time as other arrays
    #
    for ind3, kxi in enumerate(kx_arr):
        plt.scatter(B_array, final_multiarray[ind3], c=f"C{ind3}")
        plt.errorbar(B_array, final_multiarray[ind3], yerr=final_multiarray_sig[ind3], capsize=4, c=f"C{ind3}")
        plt.plot(B_array, final_multiarray[ind3],  c=f"C{ind3}", label=str(round(q(kxi), -3)))
#----------------------------------------------------------------#
        if theory:
          sample=samplename
          # theoretical values
          # extract right mode and M and gamma
          if extract_polarizers_config(exp_folder) == "EE":
              mode = 1
          elif extract_polarizers_config(exp_folder) == r"$E_{10}O$":
              mode = 2       
          else: print("Cannot extract polarizers config - mode unknown!")

          if sample == "E7":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "GCQ2":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "N19":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "N19C":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          else: print("Cannot extract sample name - M, gamma_skl unknown!")
          
          #plot:
          plt.plot(B_array, tau_theor(qperp=q(ky), qpara=q(kxi), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode), label=f"theoretical {sample}")
          # export:
          tau_theor_arr.append(np.array(tau_theor(qperp=q(ky), qpara=q(kxi), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode)))
          #----------------------------------------------------------------#

    for ind3, kxi in enumerate(kx_arr):
        xfit, yfit = np.array([]), np.array([])

        # only keep positive:
        for ind4, kk in enumerate(B_array):
            if kk > 0:
                xfit, yfit = np.append(xfit, kk), np.append(yfit, final_multiarray[ind3][ind4])

        valid_indices = ~np.isnan(xfit) & ~np.isnan(yfit)
        xfit, yfit = xfit[valid_indices], yfit[valid_indices]

        popt, pcov = curve_fit(linfit, xfit, yfit)
        fitk, fita = popt[0], popt[1]
        erfitk, erfita = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
        koef_array, offset_array = np.append(koef_array, fitk), np.append(offset_array, fita)
        er_koef_array, er_offset_array = np.append(er_koef_array, erfitk), np.append(er_offset_array, erfita)
        plt.plot(xfit[:int(len(xfit)/2)], linfit(xfit[:int(len(xfit)/2)], *popt), c="black", linestyle="--")
        fitxarr.append(xfit[:int(len(xfit)/2)])
        fityarr.append(linfit(xfit[:int(len(xfit)/2)], *popt))

    # extract data:

    name=filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}.npz"),
             B_array=B_array,#,
             final_multiarray=final_multiarray,
             final_multiarray_sig=final_multiarray_sig,
             fitx=np.array(fitxarr, dtype="object"),
             fity=np.array(fityarr, dtype="object"),
             tau_theor_arr=tau_theor_arr)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")

    plt.title(rf"Fitted correlation times 1 / $\tau$ for different $q_\parallel$'s, $k_\perp$ = {ky}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$B$ $(mT)$")
    plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend(title=r"$q_\parallel$ (1/m)", loc="upper left")
    plt.show()

    ############
    plt.clf()
    plt.errorbar(kx_arr, koef_array, yerr=er_koef_array, capsize=3)
    plt.title(rf"Fitted slopes for different $q_\parallel$'s, $k_\perp$ = {ky}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$k_\parallel$")
    plt.ylabel(r"slope $1/\tau$/$B$ [1/(s*mT)]")
    #plt.legend(title=r"$q_x$ (1/m)", loc="upper right")
    plt.show()

    name=filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}_slopes.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}_slopes.npz"),
             kx_arr=kx_arr,
             koef_array=koef_array,
             er_koef_array=er_koef_array)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")

    #############
    plt.clf()
    plt.errorbar(kx_arr, offset_array, yerr=er_offset_array, capsize=3)
    plt.title(rf"Fitted offsets for different $q_\parallel$'s, $k_\perp$ = {ky}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$k_\parallel$")
    plt.ylabel(r"offset $1/\tau$ [1/s]")
    #plt.legend(title=r"$q_x$ (1/m)", loc="upper right")
    plt.show()

    name=filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}_offsets.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_ky_{ky}_kx_{str_kx}_{samplename}_offsets.npz"),
             kx_arr=kx_arr,
             koef_array=offset_array,
             er_koef_array=er_offset_array)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")


def multimeasurement_comparison_different_qs_y_fit (FOLDER, samplename, deltat, ky_arr=[0, 1, 3], kx=0, tolerance=0.2, show_fit_plots=False, save_fit_plots = False, halldata=True, add_suptitle=r"", use_existing_fit = True, theory=False):
    """
    Perform multi-measurement comparison for one sample at different B values for different q vectors.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:
        FOLDER (str): The main folder path containing the data.

        tolerance (float, optional): The tolerance for fitting. Defaults to 0.2.
        show_fit_plots (bool, optional): Whether to show plots for fitting. Defaults to False.
        halldata (bool, optional): Whether the data is hall data or not. Defaults to True.
        add_suptitle (str, optional): Additional suptitle for the plot. Defaults to r"".
        use_existing_fit (bool, optional): Flag to indicate whether to load existing full fit data.\n
    """
    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)
    str_ky = str(ky_arr)

    for i, sample in enumerate(samplelist_full):
        if sample == samplename: # only look at one sample
            initial_settings2(exp_folder, sample)
            folderlist = folderlist_full[i]
            B_array = B_array_full[i]
            final_multiarray = np.zeros((len(ky_arr), len(B_array))) # used to store final values
            final_multiarray_sig = np.zeros((len(ky_arr), len(B_array)))  # used to store final values
            final_multiarray_ampl = np.zeros((len(ky_arr), len(B_array))) # used to store final values

            # for each q, we scan through all fields:
            for ind1, kyi in enumerate(ky_arr):
                colourbase = "#996ef1"
                percent_ini = 40
                percent_ini_copy = percent_ini
                change = False
                while np.min([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ]) < 17:
                    percent_ini += 1
                    change = True
                if change == True and i == 0:
                    print(f"Changed initial color of loading bar from {percent_ini_copy} percent to {percent_ini} percent.")
                highest = np.max([int(colourbase[1:3], 16) * (percent_ini / 100), int(colourbase[3:5], 16) * (percent_ini / 100), int(colourbase[5:7], 16) * (percent_ini / 100)  ])
                delta = (255 * percent_ini / highest - percent_ini) / len(ky_arr)
                colour = "#" + hex(int(int(colourbase[1:3], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:] + hex(int(int(colourbase[3:5], 16) * (percent_ini + delta * 0.9 * ind1) / 100))[2:]+ hex(int(int(colourbase[5:7], 16) * (percent_ini + delta * 1 * ind1) / 100))[2:]

                for ind2, SUBFOLDER in enumerate(tqdm(folderlist, ncols=100, colour=colour)): # for each B
                    mag_field = B_array[ind2]
                    curr_f = folderlist[ind2]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                    pattern = r"current_(.*?)_mA"
                    match = re.search(pattern, curr_f)
                    if match:
                        curr = match.group(1)
                    else:
                        print("No match found in the folder string.")

                    # IMPORT DATA:
                    data = np.load(SUBFOLDER + "/" +  "corr.npz")
                    t, corr = data["t"] * deltat, data["corr"]

                    datavar = np.load(SUBFOLDER + "/" +  "var.npz")
                    var1, var2 = datavar["var1"], datavar["var2"]
                    amplitude = np.abs(corr[...,0]) #* ((var1 + var2) / 2)**0.25
                    final_multiarray_ampl[ind1][ind2] = amplitude[kx][kyi]

                    # only check the desired kx, ky point:
                    try:
                        if (os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True or os.path.exists(SUBFOLDER+f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz")==True) and use_existing_fit == True:
                            try:
                                loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                                fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, kyi], fit_tau_array1[kx, kyi], sigma_tau_array1[kx, kyi], sigma_C0_array1[kx, kyi]
                            except:
                                print("")
                                datapc = np.load(
                                    SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz",
                                    allow_pickle=True)
                                popt, pcov = datapc["popt"], datapc["pcov"]
                                # fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit[ "fit_tau_array"], loaded_fit["sigma_C0_array"],loaded_fit["sigma_tau_array"]
                                fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], pcov[0][0], pcov[1][1]
                            if show_fit_plots == True or save_fit_plots == True:
                                try:
                                    try:
                                        datapc = np.load(
                                            SUBFOLDER + f"\\tmp_fit_kx{kx}_ky{kyi}_tol{tolerance}.npz",
                                            allow_pickle=True)
                                        popt, pcov = datapc["popt"], datapc["pcov"]
                                        print("using old but correct")
                                    except:
                                        datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
                                        popt, pcov = datapc["popt_2D"][int(kx), int(kyi)], datapc["pcov_2D"][int(kx), int(kyi)]
                                    plot_fit_from_existing(corr, t, kx, kyi, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}")
                                except:
                                    print("Exception showing fit plots.")
                        else:
                            fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx=kx, ky=kyi, curr=curr, mag_field=mag_field, tolerance=tolerance, showplot=show_fit_plots, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}")

                        final_multiarray[ind1][ind2] = fittau
                        final_multiarray_sig[ind1][ind2] = sigmatau
                    except:
                        print("Exception - fitting error")

    # fit for each k:
    koef_array, offset_array = np.array([]), np.array([])
    er_koef_array, er_offset_array = np.array([]), np.array([])

    def linfit(x, k, a):
        return x * k + a

    # PLOT 2D PLOT:
    fitxarr=[]
    fityarr=[]
    #-------------
    tau_theor_arr = [] # it would be more consistent if tau_theor_arr would be created simultaneoously with final_multiarray etc
    #-------------
    for ind3, kyi in enumerate(ky_arr):
        plt.scatter(B_array, final_multiarray[ind3], c=f"C{ind3}")
        plt.errorbar(B_array, final_multiarray[ind3], yerr=final_multiarray_sig[ind3], capsize=4, c = f"C{ind3}")
        plt.plot(B_array, final_multiarray[ind3],  c=f"C{ind3}", label=str(round(q(kyi), -3)))
        #----------------------------------------------------------------#
        if theory:
          sample=samplename
          # theoretical values
          # extract right mode and M and gamma
          if extract_polarizers_config(exp_folder) == "EE":
              mode = 1
          elif extract_polarizers_config(exp_folder) == r"$E_{10}O$":
              mode = 2       
          else: print("Cannot extract polarizers config - mode unknown!")

          if sample == "E7":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "GCQ2":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "N19":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          elif sample == "N19C":
              M = M_all[0]
              gamma_skl = gamma_all[0]
          else: print("Cannot extract sample name - M, gamma_skl unknown!")
          
          #plot:
          plt.plot(B_array, tau_theor(qperp=q(kyi), qpara=q(kx), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode), label=f"theoretical {sample}")
          # export:
          tau_theor_arr.append(np.array(tau_theor(qperp=q(kyi), qpara=q(kx), B=np.array(B_array), M=M, gamma_skl=gamma_skl, mode=mode)))
          #----------------------------------------------------------------#

    for ind3, kyi in enumerate(ky_arr):
        xfit, yfit = np.array([]), np.array([])

        # only keep positive:
        for ind4, kk in enumerate(B_array):
            if kk > 0: #and kk < 20:
                xfit, yfit = np.append(xfit, kk), np.append(yfit, final_multiarray[ind3][ind4])

        valid_indices = ~np.isnan(xfit) & ~np.isnan(yfit)
        xfit, yfit = xfit[valid_indices], yfit[valid_indices]

        #popt, pcov = curve_fit(linfit, xfit[15:], yfit[15:])
        popt, pcov = curve_fit(linfit, xfit, yfit)
        fitk, fita = popt[0], popt[1]
        erfitk, erfita = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
        koef_array, offset_array = np.append(koef_array, fitk), np.append(offset_array, fita)
        er_koef_array, er_offset_array = np.append(er_koef_array, erfitk), np.append(er_offset_array, erfita)
        plt.plot(xfit[:int(len(xfit)/2)], linfit(xfit[:int(len(xfit)/2)], *popt), c="black", linestyle="--")
        #plt.plot(xfit[:6], linfit(xfit[:6], *popt), c="black", linestyle="--")
        fitxarr.append(xfit[:int(len(xfit)/2)])
        fityarr.append(linfit(xfit[:int(len(xfit)/2)], *popt))

    # extract data:
    name=filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}.npz"),
             B_array=B_array,
             final_multiarray=final_multiarray,
             final_multiarray_sig=final_multiarray_sig,
             fitx=fitxarr,
             fity=fityarr,
             tau_theor_arr=tau_theor_arr)
    npz_to_csv(name, output_folder=exp_folder + "/Results")

    plt.title(rf"Fitted correlation times 1 / $\tau$ for different $q_\perp$'s, $k_\parallel$ = {kx}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$B$ $(mT)$")
    plt.ylabel(r"$1/\tau$ $(1/s)$")
    plt.legend(title=r"$q_\perp$ (1/m)", loc="upper left")
    plt.show()

    ############
    plt.clf()
    plt.errorbar(ky_arr, koef_array, yerr=er_koef_array, capsize=3)
    plt.title(rf"Fitted slopes for different $q_\perp$'s, $k_\parallel$ = {kx}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$k_\perp$")
    plt.ylabel(r"slope $1/\tau$/$B$ [1/(s*mT)]")
    #plt.legend(title=r"$q_x$ (1/m)", loc="upper right")
    plt.show()

    name=filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}_slopes.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}_slopes.npz"),
             ky_arr=ky_arr,
             koef_array=koef_array,
             er_koef_array=er_koef_array)
    npz_to_csv(name, output_folder=exp_folder + "/Results")

    #############
    plt.clf()
    plt.errorbar(ky_arr, offset_array, yerr=er_offset_array, capsize=3)
    plt.title(rf"Fitted offsets for different $q_\perp$'s, $k_\parallel$ = {kx}")
    plt.suptitle("c-DDM: " + samplename + ",  " + add_suptitle)
    plt.xlabel("$k_\perp$")
    plt.ylabel(r"offset $1/\tau$ [1/s]")
    #plt.legend(title=r"$q_x$ (1/m)", loc="upper right")
    plt.show()

    name=filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}_offsets.npz")
    np.savez(filename_update(exp_folder + f"/Results/multi_slopes_kx_{kx}_ky_{str_ky}_{samplename}_offsets.npz"),
             ky_arr=ky_arr,
             koef_array=offset_array,
             er_koef_array=er_offset_array)
    npz_to_csv(name, output_folder=exp_folder + "/Results/")


def multimeasurement_comparison_3D(FOLDER, B_target, deltat, tolerance=0.2, use_existing_fit=False, show_fit_plots=False, save_fit_plots=False, halldata=True, add_suptitle=r"$EE$ polarizers"):
    """
    Perform multi-measurement comparison of 3D plots for a target B field.

    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.

    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.

    Parameters:

    """
    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)

    for i, sample in enumerate(samplelist_full):
        print(sample)
        initial_settings2(exp_folder, sample)
        folderlist = folderlist_full[i]
        B_array = B_array_full[i]
        B_ind = closest_element_index(B_array, B_target)

        for ind, SUBFOLDER in enumerate(folderlist):
            if ind == B_ind:
                # IMPORT DATA:
                data = np.load(SUBFOLDER + "/" +  "corr.npz")
                t, corr = data["t"] * deltat, data["corr"]
                mag_field = B_array[ind]

                curr_f = folderlist[ind]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                pattern = r"current_(.*?)_mA"
                match = re.search(pattern, curr_f)
                if match:
                    curr = match.group(1)
                else:
                    print("No match found in the folder string.")

                # LOOP THROUGH THE WHOLE K-SPACE AND FIT TAU IN EVERY POINT:
                if os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True and use_existing_fit == True:
                    print("Using existing fit data.")
                    #fit_tau_array = np.load(FOLDER + f"\\fit_tau_array_tol{tolerance}.npy")
                    loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                    fit_C0_array, fit_tau_array, sigma_C0_array, sigma_tau_array = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
                    datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)

                    for kx in tqdm(range(len(corr)), desc="Fitting tau values", ncols=100, colour="#82e0aa"):
                        for ky in range(len(corr[0])):
                                fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array[kx, ky], fit_tau_array[kx, ky], sigma_tau_array[kx, ky], sigma_C0_array[kx, ky]
                                if show_fit_plots == True or save_fit_plots == True:
                                    try:
                                        popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]
                                        plot_fit_from_existing(corr, t, kx, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_3D_{str(round(mag_field,1))}mT")
                                    except:
                                        "Exception showing fit plot"
                else:
                    fit_tau_array, fit_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))
                    sigma_tau_array, sigma_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))
                    print("\n")
                    # Iterate over the correlation function's indices for fitting tau values
                    for kx in tqdm(range(len(corr)), desc="Fitting tau values", ncols=100, colour="#82e0aa"):
                        for ky in range(len(corr[0])):
                            try:
                                fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx, ky,
                                                                                tolerance=tolerance,
                                                                                showplot=show_fit_plots)
                                # Store the fitted values in the respective arrays
                                fit_C0_array[kx, ky] = fitC0
                                fit_tau_array[kx, ky] = fittau
                                sigma_C0_array[kx, ky] = sigmaC0
                                sigma_tau_array[kx, ky] = sigmatau
                            except:
                                print("Exception - fitting error")
                    np.save(SUBFOLDER + f"\\fit_tau_array_tol{tolerance}.npy", fit_tau_array)
                # PLOT 3D SURFACE PLOT OF ALL FITTED TAU VALUES:
                # have to be plotted with 2 contributions (left/ right), otherwise there is a connecting "roof"
                # have to rearange the data from [0, ... , 63, -63, -62, ... 1]
                # to [-63, ...., 0, ... 63]
                data = fit_tau_array
                x = np.fft.fftfreq(len(data), 1/len(data))
                half_index = int(len(x)/2)
                y = x[: half_index + 1]

                data_minus = data[:half_index +1 ]
                data_plus = data[half_index + 1:]
                data = np.concatenate((data_plus, data_minus))

                # take care for Nan values - replace with closest neighbour that is not Nan:
                nanarray = np.argwhere(np.isnan(data) * 1 == 1) # convert bool to int with * 1
                print(f"Start: {len(np.argwhere(np.isnan(data) * 1 == 1))} Nan values")
                for pair in nanarray:
                    xa, ya = pair[0], pair[1]
                    # find the closest working value:
                    i_updt = 1
                    while True:
                        if ya + i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya + i_updt]), axis=1)):
                            data[xa, ya] = data[xa, ya + i_updt]
                            break
                        if ya - i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya - i_updt]), axis=1)):
                            data[xa, ya] = data[xa, ya - i_updt]
                            break
                        if xa + i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa + i_updt, ya]), axis=1)):
                            data[xa, ya] = data[xa + i_updt, ya]
                            break
                        if xa - i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa - i_updt, ya]), axis=1)):
                            data[xa, ya] = data[xa - i_updt, ya]
                            break
                        i_updt += 1

                print(f"End: {len(np.argwhere(np.isnan(data) * 1 == 1))} Nan values")

                x = np.arange(min(x), max(x) + 1)
                plt.figure(figsize=(6, 6))
                ax = plt.axes(projection='3d')
                colormaps = ["Purples_r", "Greens_r", "Blues_r", "Oranges_r", "Reds_r", "Greys_r"]
                X, Y = np.meshgrid(q(x), q(y))
                ax.plot_surface(X, Y, np.transpose(data), cmap="plasma", alpha=0.3)

                ax.set_zlim([0, 4000])
                ax.set_title(r"c-DDM: " + "comparison of different samples" + "\n" + r"Fitted correlation times $1/\tau$" )
                ax.set_xlabel("$q_\parallel (1/m)$")
                ax.set_ylabel("$q_\perp (1/m)$")
                ax.set_zlabel(r"1/$\tau (1/s)$")
                ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
                ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
                ax.zaxis._axinfo["grid"]['linewidth'] = 0.1
                ax.xaxis._axinfo["grid"]['linestyle'] = "-"
                ax.yaxis._axinfo["grid"]['linestyle'] = "-"
                ax.zaxis._axinfo["grid"]['linestyle'] = "-"
    plt.show()


def multimeasurement_comparison_3D_onesample(FOLDER, B_target_list, samplename, deltat, tolerance=0.2, use_existing_fit=False, show_fit_plots=False, save_fit_plots=False, halldata=True, add_suptitle=r"$EE$ polarizers"):
    """
    Perform multi-measurement comparison of 3D plots for a target B field.
    We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
    The data must be stored in the following folder structure (current value folde names must contain numbers!):

        Experiment_folder/
        ├── Sample 1/
        │   ├── Current_value 1/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── Current_value 2/
        │   │   └── measured DDM files (corr.npz, var.npz etc)
        │   ├── ...
        │   │
        │   └── Results
        │       └── DDM_results_Hall.txt (optional)
        │
        ├── Sample 2/
        │   ├── ...
        │
        └── ...
    The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.
    Parameters:
    """
    xlabel, exp_folder, samplelist_full, folderlist_full, B_array_full = multifolder_extract(exp_folder=FOLDER, halldata=halldata)

    for i, sample in enumerate(samplelist_full):
        if sample == samplename:
            print(sample)
            initial_settings2(exp_folder, sample)
            folderlist = folderlist_full[i]
            B_array = B_array_full[i]
            for B_target in B_target_list:
                B_ind = closest_element_index(B_array, B_target)
                #print(B_target, B_ind)

                for ind, SUBFOLDER in enumerate(folderlist):

                    if ind == B_ind:
                        #print(SUBFOLDER)
                        # IMPORT DATA:
                        data = np.load(SUBFOLDER + "/" +  "corr.npz")
                        t, corr = data["t"] * deltat, data["corr"]
                        mag_field = B_array[ind]

                        curr_f = folderlist[ind]  # Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers/E7/run_0_current_0_mA
                        pattern = r"current_(.*?)_mA"
                        match = re.search(pattern, curr_f)
                        if match:
                            curr = match.group(1)
                        else:
                            print("No match found in the folder string.")

                        # LOOP THROUGH THE WHOLE K-SPACE AND FIT TAU IN EVERY POINT:
                        if os.path.exists(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz") == True and use_existing_fit == True:
                            print("Using existing fit data.")
                            #fit_tau_array = np.load(FOLDER + f"\\fit_tau_array_tol{tolerance}.npy")
                            loaded_fit = np.load(SUBFOLDER + f"\\fit_full_tol{tolerance}.npz")
                            fit_C0_array, fit_tau_array, sigma_C0_array, sigma_tau_array = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]

                            datapc = np.load(SUBFOLDER + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)

                            for kx in tqdm(range(len(corr)), desc="Fitting tau values", ncols=100, colour="#82e0aa"):
                                for ky in range(len(corr[0])):
                                        fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array[kx, ky], fit_tau_array[kx, ky], sigma_tau_array[kx, ky], sigma_C0_array[kx, ky]

                                        if show_fit_plots == True or save_fit_plots == True:
                                            try:
                                                popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]
                                                plot_fit_from_existing(corr, t, kx, ky, popt, pcov, mag_field=mag_field, curr=curr, plotshow=show_fit_plots, plotsave=save_fit_plots, out_folder=exp_folder+f"/Results/multi_3D_onesample_{str(B_target_list)}mT")
                                            except:
                                                "Exception showing fit plot"

                        else:
                            #print(SUBFOLDER)
                            fit_tau_array, fit_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))
                            sigma_tau_array, sigma_C0_array = np.zeros((len(corr), len(corr[0]))), np.zeros((len(corr), len(corr[0])))

                            print("\n")
                            # Iterate over the correlation function's indices for fitting tau values
                            for kx in tqdm(range(len(corr)), desc="Fitting tau values", ncols=100, colour="#82e0aa"):
                                for ky in range(len(corr[0])):
                                    try:
                                        fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx, ky,
                                                                                        tolerance=tolerance,
                                                                                        showplot=False)

                                        # Store the fitted values in the respective arrays
                                        fit_C0_array[kx, ky] = fitC0
                                        fit_tau_array[kx, ky] = fittau
                                        sigma_C0_array[kx, ky] = sigmaC0
                                        sigma_tau_array[kx, ky] = sigmatau
                                    except:
                                        print("Exception - fitting error")

                            np.save(SUBFOLDER + f"\\fit_tau_array_tol{tolerance}.npy", fit_tau_array)

                        # PLOT 3D SURFACE PLOT OF ALL FITTED TAU VALUES:

                        # have to be plotted with 2 contributions (left/ right), otherwise there is a connecting "roof"
                        # have to rearange the data from [0, ... , 63, -63, -62, ... 1]
                        # to [-63, ...., 0, ... 63]
                        data = fit_tau_array
                        x = np.fft.fftfreq(len(data), 1/len(data))
                        half_index = int(len(x)/2)
                        y = x[: half_index + 1]

                        data_minus = data[:half_index +1 ]
                        data_plus = data[half_index + 1:]
                        data = np.concatenate((data_plus, data_minus))

                        # take care for Nan values - replace with closest neighbour that is not Nan:
                        nanarray = np.argwhere(np.isnan(data) * 1 == 1) # convert bool to int with * 1
                        #print(f"Start: {len(np.argwhere(np.isnan(data) * 1 == 1))} Nan values")
                        for pair in nanarray:
                            xa, ya = pair[0], pair[1]
                            # find the closest working value:
                            i_updt = 1
                            while True:
                                if ya + i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya + i_updt]), axis=1)):
                                    data[xa, ya] = data[xa, ya + i_updt]
                                    break
                                if ya - i_updt in range(len(data[0])) and not np.any(np.all(nanarray == np.array([xa, ya - i_updt]), axis=1)):
                                    data[xa, ya] = data[xa, ya - i_updt]
                                    break
                                if xa + i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa + i_updt, ya]), axis=1)):
                                    data[xa, ya] = data[xa + i_updt, ya]
                                    break
                                if xa - i_updt in range(len(data)) and not np.any(np.all(nanarray == np.array([xa - i_updt, ya]), axis=1)):
                                    data[xa, ya] = data[xa - i_updt, ya]
                                    break
                                i_updt += 1

                        x = np.arange(min(x), max(x) + 1)
                        plt.figure(figsize=(6, 6))
                        ax = plt.axes(projection='3d')

                        X, Y = np.meshgrid(q(x), q(y))
                        ax.plot_surface(X, Y, np.transpose(data), cmap="plasma", alpha=0.3, label="a")

                        ax.set_zlim(0, 4000)
                        ax.set_title(r"c-DDM: " + f"comparison of different fields for {samplename}" + "\n" + r"Fitted correlation times $1/\tau$" )
                        ax.set_xlabel("$q_\parallel (1/m)$")
                        ax.set_ylabel("$q_\perp (1/m)$")
                        ax.set_zlabel(r"1/$\tau (1/s)$")
                        ax.xaxis._axinfo["grid"]['linewidth'] = 0.1
                        ax.yaxis._axinfo["grid"]['linewidth'] = 0.1
                        ax.zaxis._axinfo["grid"]['linewidth'] = 0.1
                        ax.xaxis._axinfo["grid"]['linestyle'] = "-"
                        ax.yaxis._axinfo["grid"]['linestyle'] = "-"
                        ax.zaxis._axinfo["grid"]['linestyle'] = "-"

    plt.show()

#######################################
### FUNCTIONS FOR THEORY COMPARISON ###
#######################################

def viscosity(qperp, qpara, mode, gamma1=1, a1=1, a2=1, a3=1, a4=1, a5=1, etaA=1, etaB=1, etaC=1):
    '''
    Theoretical value for viscosity.
    mode: 1 = Splay-Bend, 2=Twist-Bend # PREVERI!
    '''
    viscosity1 = gamma1 - (qperp**2 * a3 - qpara**2 * a2)**2 / (qperp**4 * etaB + qperp**2 * qpara**2 * (a1 + a3 + a4 + a5) + qpara**4 * etaC)
    viscosity2 = gamma1 - (a2 * qpara)**2 / (qperp**2 * etaA + qpara**2*etaC)
    if mode == 1:
        return viscosity1
    elif mode == 2:
        return viscosity2


def tau_theor(qperp, qpara, B, M, gamma_skl, mode, K=1, gamma1=1, a1=1, a2=1, a3=1, a4=1, a5=1, etaA=1, etaB=1, etaC=1):
    """Theoretical 1/tau value."""
    mu0 = 4 * np.pi * 10**(-7) # Vs/Am
    # B je v mT!!!
    H = B / mu0
    q = np.sqrt(qpara**2 + qperp**2)
    return 1 / 2 / viscosity(qperp, qpara, gamma1, a1, a2, a3, a4, a5, etaA, etaB, etaC, mode) * (K * q**2 + 2 * gamma_skl * mu0 * M**2 - np.sqrt(K**2 * q**4 * + 4 * gamma_skl**2 * mu0**2 * M**4) + mu0 * H * M * (1 + K * q**2 / (np.sqrt(K**2 * q**4 + 4 * gamma_skl**2 * mu0**2 * M**4 ))) )























#--------------------------obsolete stuff-------------------------------------#

#
# def multimeasurement_comparison_B_old(exp_folder, kx, ky, deltat, suffix="", description="", add_suptitle="", tolerance=0.5, halldata=False, show_fit_plots=True, showplot=True, plotsave=False, overwrite=False):
#     '''
#     Perform a comparison of measurements across different magnetic field values.
#     We choose certain kx, ky values. Then we calculate 1/tau in in this point and compare it over different B values (different folders).
#
#     We may use Hall probe results file to determine magnetic field values. If Hall file is not present, el. current values will be used.
#
#     The data must be stored in the following folder structure:
#
#         Experiment_folder/
#         ├── Sample 1/
#         │   ├── Current_value 1/
#         │   │   └── measured DDM files (corr.npz, var.npz etc)
#         │   ├── Current_value 2/
#         │   │   └── measured DDM files (corr.npz, var.npz etc)
#         │   ├── ...
#         │   │
#         │   └── Results
#         │       └── DDM_results_Hall.txt (optional)
#         │
#         ├── Sample 2/
#         │   ├── ...
#         │
#         └── ...
#     The function should be able to convert different types of folder names (decimal, non-decimal etc) to number arrays, skipping individual files and results folders.
#
#     Parameters:
#     - exp_folder (str): Path to the folder containing all samples' experiments.\n
#     - kx (float): Value of kx.\n
#     - ky (float): Value of ky.\n
#     - suffix (str, optional): Suffix of the Hall probe data filename. Defaults to "".\n
#     - description (str, optional): Description of the saved plot filename. Defaults to "".\n
#     - tolerance (float, optional): The tolerance sigmatau / fittau where 2-exp fit should be used. Defaults to 0.5 \n
#     - halldata (bool, optional): Flag to indicate whether to use Hall probe data for B values. Defaults to False.\n
#     - show_fit_plot (bool, optional): Flag to indicate whether to show the individual fit plots. Defaults to True.\n
#     - showplot (bool, optional): Flag to indicate whether to show the final plot. Defaults to True.\n
#     - plotsave (bool, optional): Flag to indicate whether to save the plot. Defaults to False.\n
#     - overwrite (bool, optional): Flag to indicate whether to overwrite existing saved plot. Defaults to False.\n
#
#     Returns:
#     - final_oneovertau_array (list): List of arrays containing the values of 1/tau for each magnetic field value.
#     '''
#     final_B_array, final_oneovertau_array, final_sigmatau_array, final_sample = [], [], [], []
#     samplelist = os.listdir(exp_folder)
#     for FOLDER in samplelist:
#         if os.path.isdir(exp_folder + "/" + FOLDER) == True and FOLDER != "results": # skip files, keep folders
#             #print(FOLDER)
#             sample = os.path.basename(FOLDER)
#             initial_settings(exp_folder, sample)
#             FOLDER = exp_folder + "/" + FOLDER
#
#             oneovertau_array = np.array([])
#             sigmatau_array = np.array([])
#             B_array = np.array([]) # for current or magnetic field values
#             xlabel = r"I [mA] (1 A $\approx$ 30 mT)"
#
#             # find all folders and make B array from them:
#             folderlist = os.listdir(FOLDER)
#             for i in range(len(folderlist)):
#                 if os.path.isdir(FOLDER + "/" + folderlist[i]) == True and folderlist[i] != "results": # skip files, keep folders
#                     # create an array with B values from each experiment:
#                     try: # decimal number exctraction:
#                         B_array = np.append(B_array, float(re.findall("\d+\.\d+", folderlist[i])[0]))
#                     except: # non-decimal number exctraction:
#                         B_array = np.append(B_array, float(re.findall(r"\d+", folderlist[i])[0]))
#                     folderlist[i] = FOLDER + '/' + str(folderlist[i]) # make absolute path
#                 else: folderlist[i] = None
#             folderlist = [x for x in folderlist if x is not None] # keep only folders
#
#             if halldata == True: # Update B array with B values, calculated from Hall probe
#                 current, volt_hall = np.loadtxt(FOLDER+"/results/DDM_results_Hall" + suffix + ".txt", delimiter=",", unpack=True)
#                 if len(volt_hall) == len(B_array): # if something is wrong, just use current values
#                     xlabel = "B [mT]"
#                     B_array = np.array([])
#                     for volt in volt_hall:
#                         B_array = np.append(B_array, magnetic_field(volt/1000))
#                 else: print("Legnth mismatch. Using current values instead.")
#
#             for SUBFOLDER in folderlist:
#                 # analysis
#                 #print(SUBFOLDER[len(FOLDER):])
#                 data = np.load(SUBFOLDER + "/" +  "corr.npz")
#                 t, corr = data["t"] * deltat, data["corr"]
#                 fitC0, fittau, sigmatau, sigmaC0 = fit_corr(corr, t, kx, ky, tolerance=tolerance, showplot=show_fit_plots, plotsave=False)
#                 oneovertau_array = np.append(oneovertau_array, fittau)
#                 sigmatau_array = np.append(sigmatau_array, sigmatau)
#
#             # update final arrays:
#             final_B_array.append(B_array)
#             final_oneovertau_array.append(oneovertau_array)
#             final_sigmatau_array.append(sigmatau)
#             final_sample.append(sample)
#
#     # plot final arrays:
#     plt.clf()
#     for i in range(len(final_B_array)):
#         plt.errorbar(final_B_array[i], final_oneovertau_array[i], yerr=final_sigmatau_array[i], capsize=3, label=final_sample[i])
#
#     plt.xlabel(xlabel)
#     plt.ylabel(r"1/$\tau$[1/s]")
#     plt.suptitle(r"$\tau^{-1}(B)$ dependence" + " - " + add_suptitle)
#     #plt.suptitle(r"$\tau^{-1}(B)$ dependence - uncrossed polarizers")
#     plt.title(f"($\parallel$ = {kx}, $k_\perp$ = {ky}) point")
#     plt.legend()
#
#     if plotsave == True:
#         initial_settings(exp_folder, description)
#         save_figure("multimeasurement", overwrite=overwrite)
#
#     if showplot == True:
#         plt.show()
#
#     return final_oneovertau_array




# def plot_corr_der(corr, t, kx, ky, plotsave=False, overwrite=False):
#     """
#     Plots the correlation function for a given (kx, ky) point.
#       Also plot 2nd order deriv zeros
#
#     Parameters:
#         corr: Correlation functions data
#         kx (int): The k_x value.\n
#         ky (int): The k_y value.\n
#         plotsave (bool): Whether to save the plot. \n
#         overwrite (bool): Whether to overwrite existing file.
#
#     """
#     kx_plot = np.fft.fftfreq(len(corr), 1/len(corr))[kx] # sort correctly just for plot label (the whole array is sorted in later steps). Indexing works anyway, because fftfreq [-kx] = - kx.
#
#     #plt.plot(np.log(t), np.abs(corr[kx, ky]), label=rf"$k_x$={kx_plot}, $k_y$={ky}")
#     #plt.plot(np.log(t), np.gradient(np.abs(corr[kx, ky]), np.log(t)), label=rf"$k_x$={kx_plot}, $k_y$={ky} grad")
#     #plt.plot(np.log(t), np.gradient(np.gradient(np.abs(corr[kx, ky]), np.log(t)), np.log(t)), label=rf"$k_x$={kx_plot}, $k_y$={ky} grad 2")
#     #plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"$k_x$={kx_plot}, $k_y$={ky}")
#     plt.plot(np.log10(t), np.abs(corr[kx, ky]), label=rf"$k_\parallel$={kx_plot}, $k_\perp$={ky}")
#
#     # 2-exp fit option:
#     def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
#         if f > f1:  # first must be larger!
#             return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
#         else:
#             return np.inf
#     try:
#         # if one-exp fit doesnt work, use two-exp:
#         twoexp = True
#         popt, pcov = curve_fit(fit_func2, t[7:], corr[kx, ky][7:],
#                                p0=[300, 1, 1, 1],
#                                # p0=[10, 1, 1, 0.001],
#                                bounds=([1, - np.inf, - np.inf, - np.inf], [3000, np.inf, np.inf,
#                                                                            100]))  # we set initial estimations so that the roles of both exponent terms remain the same.
#         fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
#         fitC1, fittau2 = popt[2], popt[3]  # for initial parameters loop
#
#         if sigmatau / fittau > 1:  # tau = 1/tau
#             fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#         # add: if fittau = weird -> np.nan (check typical values EE/E10O, < 4000 ish )
#         # test saving size:
#
#     except:
#         #print("Could not perform fit.")
#         fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#
#     y = fit_func2(t,*popt)
#     plt.plot(np.log10(t), y, linestyle="--", c="black", label = "fit:\n" + rf"$C_0$ = {round(popt[1],3)} $\pm$ {round(np.sqrt(pcov[1,1]),3)}"
#                              + "\n" +
#                              rf"$1/\tau$ = {round(popt[0], 5)} /s $\pm$ {round(np.sqrt(pcov[0,0]), 5)} /s"
#                              + "\n" +
#                              r"$1/\tau_2$ = " + f"{round(popt[3], 6)} /s  $\pm$ {round(np.sqrt(pcov[3,3]), 7)} /s")
#     plt.plot(np.log10(t), np.gradient(np.abs(y), np.log10(t)), label=rf"$k_\parallel$={kx_plot}, $k_\perp$={ky} grad")
#     plt.plot(np.log10(t), np.gradient(np.gradient(np.abs(y), np.log10(t)), np.log10(t)), label=rf"$k_\parallel$={kx_plot}, $k_\perp$={ky} grad 2")
#     grad2 = np.gradient(np.gradient(np.abs(y), np.log10(t)), np.log10(t))
#
#     sign = np.sign(grad2)
#     dif = np.diff(sign)
#
#     crossings_plus = np.where((dif == 2) | (dif == 1))[0]
#     print(crossings_plus)
#
#     #plt.plot(np.log10(t)[:-1], dif)
#
#     if len(crossings_plus) == 2:
#         p1, p2 = [np.log10(t)[crossings_plus[0]], grad2[crossings_plus[0]]],  [np.log10(t)[crossings_plus[1]], grad2[crossings_plus[1]]]
#         plt.scatter(p1[0],p1[1], s=50, label = f"{round(1/10**p1[0],1)}")
#         plt.scatter(p2[0],p2[1], s=50, label = f"{round(1/10**p2[0],1)}")
#
#     if len(crossings_plus) == 1:
#         p1 = [np.log10(t)[crossings_plus[0]], grad2[crossings_plus[0]]]
#         plt.scatter(p1[0],p1[1], s=50, label = f"{round(1/10**p1[0],1)}")
#         #plt.scatter(p2[0],p2[1], s=50)
#
#
#
#
#
#     plt.title(rf"Correlation function, $k_\parallel$={kx_plot}, $\perp$={ky}")
#     plt.suptitle("c-DDM: " + SAMPLE)
#     plt.xlabel("time (s)")
#     plt.ylabel("correlation")
#     plt.legend()
#
#     if plotsave == True:
#         save_figure(f"corr_func_{kx_plot}_{ky}", overwrite=overwrite)
#
#     plt.show()


# def fit_corr_der(corr, t, kx, ky, tolerance=0.03, init_pars=[300, 1, 1, 1], constraints=([1, 0, 0, 0], [3000, 10, 10, 1]), deriv_estimate=True, showplot = False, plotsave=False, overwrite=False, old_return=True):
#     """
#     Fits the correlation function in a given (kx, ky) point.
#     deriv_estimate routine:
#       Because the 2-exponential fitting can be misleading (we see a perfect fit but because the roles
#       of two exponents get "mixed", the fitted values can be very wrong), we have to use bounds for parameters.
#       Because the bounds can vary greatly beteween points in k-space, materials etc, we estimate them
#       by finding the roots of the 2nd order derivative of the smoothened data.
#       Smoothing is done by applying an unconstrained fit, only to get a smooth function for calculating derivatives.
#       If this works, initial parameters and parameter bounds are set around these two points.
#
#     If the error is large enough, fitting with two exponential functions is used.
#
#     FOR USE WITH DIFFERENT MATERIALS: MANAULLY ADAPT FIT BOUNDS!
#
#     Parameters:
#         kx (int): The k_x value.\n
#         ky (int): The k_y value.\n
#         tolerance (float): The tolerance sigmatau / fittau where 2-exp fit should be used \n
#         init_pars: initial parameters \n
#         constraints: lower and upper bounds for tau1, C1, C2, tau2 \n
#         deriv_estimate: Wheter to use initial estimate and bounds guessing with 2nd order derivative
#         showplot (bool): Whether to show the plot of the fit (default is False). \n
#         plotsave (bool): Whether to save the plot (default is False). \n
#         overwrite (bool): Whether to overwrite existing file (default is False).\n
#         old_return(bool): Wheter to return 4 params (True, default) or 6 (with C2, tau2) + popt and pcov\n
#
#     Returns:
#         tuple: A tuple containing the fitC0, fittau(=1/tau), sigmatau, and sigmaC0 values.
#     """
#
#     # FIRST: UNCONSTRAINED FIT TO GET A SMOOTH FUNCTION
#     if deriv_estimate == True:
#
#
#         def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
#             if f > f1:  # first must be larger!
#                 return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
#             else:
#                 return np.inf
#
#
#         try:
#             twoexp = True
#             popt, pcov = curve_fit(fit_func2, t[7:], corr[kx, ky][7:],
#                                   p0=init_pars,
#                                   bounds=([1, 0, 0, 0], [3000, 10, 10, 100]))
#             fitC0, fittau, sigmatau, sigmaC0, fittau2 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1]), popt[3]
#             success = True
#             #print(fittau, fittau2)
#
#             if sigmatau / fittau > 1: # unreliable fit
#               fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#               success = False
#
#
#         except:
#             fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#             success = False
#             # if np.nan, dont use that ...
#
#         if success:
#             # find the roots of 2nd order derivative. They should be very close to tau 1 and tau 2. Use them for initital estimates and constraints for actual fit
#             y = fit_func2(t, *popt)
#             grad2 = np.gradient(np.gradient(np.abs(y), np.log10(t)), np.log10(t))
#             sign = np.sign(grad2)
#             dif = np.diff(sign)
#             crossings_plus = np.where((dif == 2) | (dif == 1))[0]
#             #print(crossings_plus)
#             #print(len(crossings_plus))
#             # if one crossing: use 1-exp!
#             if len(crossings_plus) == 1:
#                 tau1_est = 1/10**(np.log10(t)[crossings_plus[0]]) # put back in exp
#                 # set init pars and constraints, otherwise (success == False) use default ones:
#                 init_pars = [tau1_est, 1, 1, 0.00001]
#                 constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, 0.1])
#                 #print(constraints)
#                 #tolerance = 100000  # = use 1-exp!
#
#             # if two crossings:
#             if len(crossings_plus) == 2:
#                 tau1_est, tau2_est = 1/10**(np.log10(t)[crossings_plus[0]]), 1/10**(np.log10(t)[crossings_plus[1]]) # put back in exp
#                 #print(tau1_est, tau2_est)
#                 if tau2_est < 10:
#                     # set init pars and constraints, otherwise (success == False) use default ones:
#                     init_pars = [tau1_est, 1, 1, tau2_est]
#                     if tau2_est < 1:
#                         constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, tau2_est + 0.5])
#                     else:
#                         constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, tau2_est + 2])
#
#     # ACTUAL FIT:
#     kx_plot = np.fft.fftfreq(len(corr), 1/len(corr))[kx] # sort correctly just for plot label (the whole array is sorted in later steps). Indexing works anyway, because fftfreq [-kx] = - kx.
#     twoexp = False
#
#
#     def fit_func(t, f, C0):  # f = 1 / tau
#          return C0 * np.exp(- f * t)
#
#
#     # 2-exp fit option:
#     def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
#         if f > f1: # first must be larger!
#             return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
#         else:
#             return np.inf
#
#     try:
#         # try with one exponent:
#         popt, pcov = curve_fit(fit_func, t, corr[kx, ky])
#         fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
#
#         # if one-exp fit doesnt work, use two-exp:
#         if sigmatau / fittau > tolerance:
#             twoexp = True
#             #print(constraints)
#             popt, pcov = curve_fit(fit_func2, t, corr[kx, ky],
#                                    p0=init_pars,
#                                    #p0=[10, 1, 1, 0.001],
#                                    bounds=constraints) # we set initial estimations so that the roles of both exponent terms remain the same.
#             fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
#             fitC1, fittau2 = popt[2], popt[3] # for initial parameters loop
#
#         if sigmatau / fittau > 1: # tau = 1/tau; unreliable fits
#             fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#
#     except:
#         #print("Could not perform fit.")
#         fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#
#
#     if showplot == True:
#         try:
#             plt.cla()
#             plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"data, $k_\parallel$ = {kx_plot}, $k_\perp$ = {ky}")
#
#             if twoexp == True:
#                 plt.semilogx(t, fit_func2(t, *popt), c="black", linestyle="--",
#                              label = "fit:\n" + rf"$C_0$ = {round(popt[1],3)} $\pm$ {round(np.sqrt(pcov[1,1]),3)}"
#                              + "\n" +
#                              rf"$1/\tau$ = {round(popt[0], 5)} /s $\pm$ {round(np.sqrt(pcov[0,0]), 5)} /s"
#                              + "\n" +
#                              r"$1/\tau_2$ = " + f"{round(popt[3], 6)} /s  $\pm$ {round(np.sqrt(pcov[3,3]), 7)} /s")
#             else:
#                 plt.semilogx(t, fit_func(t, *popt), c="black", linestyle="--",
#                              label = rf"fit, $C_0$ = {round(popt[1],3)} $\pm$ {round(np.sqrt(pcov[1,1]),3)}"
#                              + "\n" +
#                              rf"      $1/\tau$ = {round(popt[0], 5)} /s $\pm$ {round(np.sqrt(pcov[0,0]), 5)} /s")
#
#
#             plt.title("Correlation function")
#             plt.suptitle("c-DDM: " + SAMPLE)
#             plt.xlabel("time (s)")
#             plt.ylabel("correlation")
#             plt.legend()
#
#             if plotsave == True:
#                 save_figure(f"corr_func_fit_{kx_plot}_{ky}", overwrite=overwrite)
#             plt.show()
#
#         except:
#             print("Plotting the correlation function was not possible.")
#
#     if old_return==True:
#         return fitC0, fittau, sigmatau, sigmaC0
#     else:
#         return fitC0, fittau, sigmatau, sigmaC0, fitC1, fittau2, popt, pcov


# def fit_corr_delete_justforlines(corr, t, kx, ky, tolerance=0.03, init_pars=[300, 1, 1, 1], constraints=([1, 0, 0, 0], [3000, 10, 10, 1]),
#              deriv_estimate=False, showplot=False, plotsave=False, overwrite=False, old_return=True):
#     """
#     Fits the correlation function in a given (kx, ky) point.
#     deriv_estimate routine:
#       Because the 2-exponential fitting can be misleading (we see a perfect fit but because the roles
#       of two exponents get "mixed", the fitted values can be very wrong), we have to use bounds for parameters.
#       Because the bounds can vary greatly beteween points in k-space, materials etc, we estimate them
#       by finding the roots of the 2nd order derivative of the smoothened data.
#       Smoothing is done by applying an unconstrained fit, only to get a smooth function for calculating derivatives.
#       If this works, initial parameters and parameter bounds are set around these two points.
#
#     If the error is large enough, fitting with two exponential functions is used.
#
#     FOR USE WITH DIFFERENT MATERIALS: MANAULLY ADAPT FIT BOUNDS!
#
#     Parameters:
#         kx (int): The k_x value.\n
#         ky (int): The k_y value.\n
#         tolerance (float): The tolerance sigmatau / fittau where 2-exp fit should be used \n
#         init_pars: initial parameters \n
#         constraints: lower and upper bounds for tau1, C1, C2, tau2 \n
#         deriv_estimate: Wheter to use initial estimate and bounds guessing with 2nd order derivative
#         showplot (bool): Whether to show the plot of the fit (default is False). \n
#         plotsave (bool): Whether to save the plot (default is False). \n
#         overwrite (bool): Whether to overwrite existing file (default is False).\n
#         old_return(bool): Wheter to return 4 params (True, default) or 6 (with C2, tau2) + popt and pcov\n
#
#     Returns:
#         tuple: A tuple containing the fitC0, fittau(=1/tau), sigmatau, and sigmaC0 values.
#     """
#
#     # FIRST: UNCONSTRAINED FIT TO GET A SMOOTH FUNCTION
#     if deriv_estimate == True:
#
#         def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
#             if f > f1:  # first must be larger!
#                 return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
#             else:
#                 return np.inf
#
#         try:
#             twoexp = True
#             popt, pcov = curve_fit(fit_func2, t[7:], corr[kx, ky][7:],
#                                    p0=init_pars,
#                                    bounds=([1, 0, 0, 0], [3000, 10, 10, 200]))
#             fitC0, fittau, sigmatau, sigmaC0, fittau2 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1]), \
#                                                         popt[3]
#             success = True
#             # print(fittau, fittau2)
#
#             if sigmatau / fittau > 1:  # unreliable fit
#                 fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#                 success = False
#
#
#         except:
#             fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#             success = False
#             # if np.nan, dont use that ...
#
#         if success:
#             # find the roots of 2nd order derivative. They should be very close to tau 1 and tau 2. Use them for initital estimates and constraints for actual fit
#             y = fit_func2(t, *popt)
#             grad2 = np.gradient(np.gradient(np.abs(y), np.log10(t)), np.log10(t))
#             sign = np.sign(grad2)
#             dif = np.diff(sign)
#             crossings_plus = np.where((dif == 2) | (dif == 1))[0]
#             # print(crossings_plus)
#             # print(len(crossings_plus))
#             # if one crossing: use 1-exp!
#             if len(crossings_plus) == 1:
#                 print("one")
#                 tau1_est = 1 / 10 ** (np.log10(t)[crossings_plus[0]])  # put back in exp
#                 # set init pars and constraints, otherwise (success == False) use default ones:
#                 init_pars = [tau1_est, 1, 1, 0.00001]
#                 constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, 0.1])
#                 # print(constraints)
#                 # tolerance = 100000  # = use 1-exp!
#
#             # if two crossings:
#             if len(crossings_plus) == 2:
#                 tau1_est, tau2_est = 1 / 10 ** (np.log10(t)[crossings_plus[0]]), 1 / 10 ** (
#                 np.log10(t)[crossings_plus[1]])  # put back in exp
#                 # print(tau1_est, tau2_est)
#                 print(t[crossings_plus[1]])
#                 if tau2_est < 10:
#                     # set init pars and constraints, otherwise (success == False) use default ones:
#                     init_pars = [tau1_est, 1, 1, tau2_est]
#                     if tau2_est < 1:
#                         constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, tau2_est + 0.5])
#                     else:
#                         constraints = ([tau1_est / 5, 0, 0, 0], [4000, 10, 10, tau2_est + 4])
#
#     # ACTUAL FIT:
#     kx_plot = np.fft.fftfreq(len(corr), 1 / len(corr))[
#         kx]  # sort correctly just for plot label (the whole array is sorted in later steps). Indexing works anyway, because fftfreq [-kx] = - kx.
#     twoexp = False
#
#     def fit_func(t, f, C0):  # f = 1 / tau
#         return C0 * np.exp(- f * t)
#
#     # 2-exp fit option:
#     def fit_func2(t, f, C0, C1, f1):  # f = 1 / tau
#         if f > f1:  # first must be larger!
#             return C0 * np.exp(- f * t) + C1 * np.exp(- f1 * t)
#         else:
#             return np.inf
#
#     try:
#         # try with one exponent:
#         popt, pcov = curve_fit(fit_func, t, corr[kx, ky])
#         fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
#
#         # if one-exp fit doesnt work, use two-exp:
#         if sigmatau / fittau > tolerance:
#             twoexp = True
#             # print(constraints)
#             popt, pcov = curve_fit(fit_func2, t, corr[kx, ky],
#                                    p0=init_pars,
#                                    # p0=[10, 1, 1, 0.001],
#                                    bounds=constraints)  # we set initial estimations so that the roles of both exponent terms remain the same.
#             fitC0, fittau, sigmatau, sigmaC0 = popt[1], popt[0], np.sqrt(pcov[0, 0]), np.sqrt(pcov[1, 1])
#             fitC1, fittau2 = popt[2], popt[3]  # for initial parameters loop
#
#         if sigmatau / fittau > 1:  # tau = 1/tau; unreliable fits
#             fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#
#     except:
#         # print("Could not perform fit.")
#         fitC0, fittau, sigmatau, sigmaC0 = np.nan, np.nan, np.nan, np.nan
#
#     if showplot == True:
#         try:
#             plt.cla()
#             plt.semilogx(t, np.abs(corr[kx, ky]), label=rf"data, $k_\parallel$ = {kx_plot}, $k_\perp$ = {ky}")
#
#             if twoexp == True:
#                 plt.semilogx(t, fit_func2(t, *popt), c="black", linestyle="--",
#                              label="fit:\n" + rf"$C_0$ = {round(popt[1], 1)} $\pm$ {round(np.sqrt(pcov[1, 1]), 1)}"
#                                    + "\n" +
#                                    rf"$1/\tau$ = {round(popt[0], 1)} /s $\pm$ {round(np.sqrt(pcov[0, 0]), 1)} /s"
#                                    + "\n" +
#                                    r"$1/\tau_2$ = " + f"{round(popt[3], 1)} /s  $\pm$ {round(np.sqrt(pcov[3, 3]), 1)} /s")
#             else:
#                 plt.semilogx(t, fit_func(t, *popt), c="black", linestyle="--",
#                              label=rf"fit, $C_0$ = {round(popt[1], 3)} $\pm$ {round(np.sqrt(pcov[1, 1]), 3)}"
#                                    + "\n" +
#                                    rf"      $1/\tau$ = {round(popt[0], 5)} /s $\pm$ {round(np.sqrt(pcov[0, 0]), 5)} /s")
#
#             if deriv_estimate:
#                 if len(crossings_plus) == 2:
#                     print(crossings_plus)
#                     # p1, p2 = [np.log10(t)[crossings_plus[0]], grad2[crossings_plus[0]]], [np.log10(t)[crossings_plus[1]],
#                     # plt.scatter(t[crossings_plus[0]], 0, s=50,
#                     #            label=f"{round(1/t[crossings_plus[0]], 1)}")  # grad2[crossings_plus[1]]]
#                     # plt.scatter(t[crossings_plus[1]], 0, s=50,
#                     #            label=f"{round(1/t[crossings_plus[1]], 1)}")
#                     plt.axvline(x=t[crossings_plus[0]],
#                                 label=f"2nd order derivative roots: \n1/t = {round(1 / t[crossings_plus[0]], 1)} 1/s",
#                                 c="C3", linestyle="-.")
#                     plt.axvline(x=t[crossings_plus[1]], label=f"1/t = {round(1 / t[crossings_plus[1]], 1)} 1/s", c="C3",
#                                 linestyle=":")
#                     # plt.scatter(10 ** p1[0], 0, s=50, label=f"{round(1 / 10 ** p1[0], 1)}")
#                     # plt.scatter(10 ** p2[0], 0, s=50, label=f"{round(1 / 10 ** p2[0], 1)}")
#
#                 if len(crossings_plus) == 1:
#                     plt.axvline(x=t[crossings_plus[0]],
#                                 label=f"2nd order derivative roots: \n 1/t = {round(1 / t[crossings_plus[0]], 1)} 1/s",
#                                 c="C3",
#                                 linestyle="-.")
#
#                     # plt.scatter(t[crossings_plus[0]], 0, s=50,
#                     #            label=f"{round(1/t[crossings_plus[0]], 1)}")
#                     # plt.scatter(p2[0],p2[1], s=50)
#
#             plt.title("Correlation function")
#             plt.suptitle("c-DDM: " + SAMPLE)
#             plt.xlabel("time (s)")
#             plt.ylabel("correlation")
#             plt.legend()
#
#             if plotsave == True:
#                 save_figure(f"corr_func_fit_{kx_plot}_{ky}", overwrite=overwrite)
#             plt.show()
#
#         except:
#             print("Plotting the correlation function was not possible.")
#
#     if old_return == True:
#         return fitC0, fittau, sigmatau, sigmaC0  # ORIGINAL
#         # return fitC0, fittau2, sigmatau/10000, sigmaC0
#     else:
#         return fitC0, fittau, sigmatau, sigmaC0, fitC1, fittau2, popt, pcov  # ORIGINAL
#         # return fitC0, fittau2, sigmatau/10000, sigmaC0, fitC1, fittau2, popt, pcov












# FIND NICE PLOTS:

# array_value_func = np.array([])
# array_kx = np.array([])
# array_ky = np.array([])
# for kx in tqdm(range(75), ncols=100):
#     for ky in range(75):
#         try:
#             data = multimeasurement_comparison_B("D:/Users Data/Simon/Magnetic experiments/test runs manual NEW samples_2_para", kx=kx, ky=ky, halldata=False, showplot=False)
#             array1 = np.array(data[0])
#             average1 = np.mean(array1)
#             array2 = np.array(data[1])
#             average2 = np.mean(array2)
#             array3 = np.array(data[2])
#             average3 = np.mean(array3)
#             value_func = 0
#             for element in array1:
#                 value_func += (element - average1)**2
#             for element in array2:
#                 value_func += (element - average2)**2
#             for element in array3:
#                 value_func += (element - average3)**2
#             array_value_func = np.append(array_value_func, value_func)
#             array_kx, array_ky = np.append(array_kx, kx), np.append(array_ky, ky)
#         except:
#             pass
#             #print("Could not perform fitting.")

# np.save("save_val.npy", array_value_func)
# np.save("save_kx.npy", array_kx)
# np.save("save_ky.npy", array_ky)
# array_value_func = np.load("save_val.npy")
# array_kx = np.load("save_kx.npy")
# array_ky = np.load("save_ky.npy")


# array_value_func_sorted_ind = np.argsort(array_value_func)
# print(array_value_func_sorted_ind)

# for ind in array_value_func_sorted_ind[:5]:
#     kx_sort, ky_sort = int(array_kx[ind]), int(array_ky[ind])
#     print(ind)
#     print(kx_sort)
#     multimeasurement_comparison_B("D:/Users Data/Simon/Magnetic experiments/test runs manual NEW samples_2_para", kx=kx_sort, ky=ky_sort, halldata=False, showplot=True)



