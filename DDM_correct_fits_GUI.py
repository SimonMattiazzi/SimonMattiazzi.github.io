# -*- coding: utf-8 -*-
"""
This is a GUI to select the plots of wrongly fitted
autocorrelation plots from a folder, manually tweak fitting
parameters to achive a good fit
and overwrite stored data.
"""
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
import re
import numpy as np
import DDM_analysis_module_Simon_y0 as DDM

# INITIAL SETTINGS:
expfolder = "D:/Users Data/Simon/Magnetic experiments/Automatic/Run 2/EE polarizers"
tolerance = 0.07
deltat = 110/1000000 # in seconds
correct_but_old = False # to use if you are "correcting" unsaved values - that is where you actually save them (SET THIS TO FALSE IF "fit_full_....npz" and "popt_pcov_2D.npz" exist)
cutoff = 0.6

def process(showold, refit, rewrite_and_continue, swaptau=False, only_continue=False):
    global index
    # choose correct filename and update the "fit i out of N" label
    update_label_text()
    filename=file_names[index]

    if only_continue: # skip and go to next graph without anything else
        index += 1
        # clear graph2:
        plt.clf()
        canvas2.draw()
    
    else: 
        # extract relevant information from filename:
        pattern = r"corr_func_fit_(\w+)_(.*?)_mA_(.*?)_mT_kx(-?\d+)_ky(-?\d+)(_\d+)?.png" 
        match = re.search(pattern, filename)
        if match:
            sample = match.group(1)
            current_str = match.group(2)
            magnetic_field_str = match.group(3)
            kx = int(match.group(4))
            ky = int(match.group(5))
            print(sample, current_str, magnetic_field_str, kx, ky)
        else:
            print("No match found in the filename.")
            
        # FIND THE CORRECT FOLDER by checking the current value
        samplefolder = expfolder + f"/{sample}"
        pattern2 = rf"(.+)_({current_str})_mA"
        count = 0
        for folders in os.listdir(samplefolder):
            match = re.search(pattern2, folders)
            if match:
                count += 1
                final_folder = samplefolder + f"/{match[0]}"
                print(final_folder)
            if count > 1:
                print("More than one current match!")
        
        # step two: LOAD existing FIT 
        
        data = np.load(final_folder + "/" +  "corr.npz")
        t, corr = data["t"] * deltat, data["corr"]

        try:
            loaded_fit = np.load(final_folder + f"/fit_full_tol{tolerance}.npz")
            fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
            fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky]
        except:
            print("No fit file")
        try:
            datapc = np.load(final_folder + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
            popt2D, pcov2D = datapc["popt_2D"], datapc["pcov_2D"]
            popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]
        except:
            print("")

        # show old fit:
        if showold:
            try:
                DDM.initial_settings(folder=final_folder, sample=sample, deltat_=deltat)
                print("here")
                DDM.plot_fit_from_existing(corr, t, kx=kx, ky=ky, popt=popt, pcov=pcov, mag_field=float(magnetic_field_str), curr=float(current_str), plotshow=True, canvas10=True, deriv=True)
            except:
                print("No such fit file!")
                DDM.fit_corr(corr, t, kx=kx, ky=ky, tolerance=tolerance, mag_field=float(magnetic_field_str), curr=float(current_str), showplot=True, plotshow=False, canvas1=True)
                plt.legend()
                canvas1.draw()
        # try new fit with input parameter:
        if refit:
            print(entry.get())
            tau2upperbound = float(entry.get())
            #print(entry2.get())
            #if entry2.get() == None:
            #    tolerance2 = tolerance
            #else:
            #    tolerance2 = entry2.get()
            #    print(tolerance2)
            #tolerance2 = 0.0001
            tolerance2 = float(entry2.get())
            print(tolerance2)

            cutoff2 = float(entry3.get())
            print(cutoff2)

            tau2lowerbound = float(entry4.get())
            print(tau2lowerbound)

            if swaptau: # use tau2 as tau1 in case of the "bump"
                fitC1_new, fittau2_new, sigmatau_new, sigmaC0_new, fitC0_new, fittau_new, popt_new, pcov_new = DDM.fit_corr(corr, t, kx, ky, cutoff=cutoff2, bounds=([1, 0, 0, tau2lowerbound, 0], [3000, 1, 1, tau2upperbound, 1]), showplot=True, old_return=False, canvas1=True)
                popt_new[0], popt_new[3] = popt_new[3], popt_new[0]
                popt_new[1], popt_new[2] = popt_new[2], popt_new[1]
                pcov_new[0][0], pcov_new[3][3] = pcov_new[3][3], pcov_new[0][0]
                pcov_new[1][1], pcov_new[2][2] = pcov_new[2][2], pcov_new[1][1]
            else:
                try:
                    #print("here2")
                    fitC0_new, fittau_new, sigmatau_new, sigmaC0_new, fitC1_new, fittau2_new, popt_new, pcov_new = DDM.fit_corr(corr, t, kx, ky, cutoff=cutoff2, bounds=([1, 0, 0, tau2lowerbound, 0], [3000, 1, 1, tau2upperbound, 1]), tolerance=tolerance2, showplot=True, plotshow=True, old_return=False, canvas1=True)
                except: #oneexp
                    #print("onexp")
                    fitC0_new, fittau_new, sigmatau_new, sigmaC0_new,  popt_new, pcov_new = DDM.fit_corr(
                        corr, t, kx, ky, cutoff=cutoff2, bounds=([1, 0, 0, tau2lowerbound, 0], [3000, 1, 1, tau2upperbound, 1]),tolerance=tolerance2, showplot=True, plotshow=True, old_return=False, canvas1=True)

            # use this value and overwrite the old fit (old one changes name to "..._old_i")
            if rewrite_and_continue:
                #
                if correct_but_old:

                    print(f"saving ... {final_folder}/tmp_fit_kx{kx}_ky{ky}.npz")
                    np.savez(final_folder + f"/tmp_fit_kx{kx}_ky{ky}_tol{tolerance}.npz",
                             popt=popt_new,
                             pcov=pcov_new)
                #
                else:
                    # save old fit with new name: "_old_i"
                    print("Saving old ones with old_i filename ...")
                    np.savez(DDM.filename_update(final_folder + f"/popt_pcov_2D_tol{tolerance}_old.npz"),
                            popt_2D=popt2D,
                            pcov_2D=pcov2D)

                    np.savez(DDM.filename_update(final_folder + f"/fit_full_tol{tolerance}_old.npz"),
                              fit_C0_array = fit_C0_array1,
                              fit_tau_array = fit_tau_array1,
                              sigma_C0_array = sigma_C0_array1,
                              sigma_tau_array = sigma_tau_array1
                              )

                    # save new fit by inputing new numbers into old array:
                    try: #if y0 exists
                        print("Saving new, overwriting original filename, y0 ...")
                        popt2D[int(kx), int(ky)], pcov2D[int(kx), int(ky)] = popt_new, pcov_new # replace values
                        np.savez(final_folder + f"/popt_pcov_2D_tol{tolerance}.npz",
                                popt_2D=popt2D,
                                pcov_2D=pcov2D)
                        # shrani se uni drugi seznam ...
                        fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[
                            kx, ky] = fitC0_new, fittau_new, sigmatau_new, sigmaC0_new
                        np.savez(final_folder + f"/fit_full_tol{tolerance}.npz",
                                 fit_C0_array=fit_C0_array1,
                                 fit_tau_array=fit_tau_array1,
                                 sigma_C0_array=sigma_C0_array1,
                                 sigma_tau_array=sigma_tau_array1
                                 )

                        # if twoexp:
                        #     return fitC0, fittau, sigmatau, sigmaC0, fitC1, fittau2, popt, pcov  # original
                        # else:
                        #     return fitC0, fittau, sigmatau, sigmaC0, popt, pcov  # original

                        # fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky], fit_y0_array1[kx, ky], sigma_y0_array1[kx, ky] = fitC0_new, fittau_new, sigmatau_new, sigmaC0_new, fity0_mew, sigmay0_new
                        # np.savez(final_folder + f"/fit_full_tol{tolerance}.npz",
                        #           fit_C0_array = fit_C0_array1,
                        #           fit_tau_array = fit_tau_array1,
                        #           sigma_C0_array = sigma_C0_array1,
                        #           sigma_tau_array = sigma_tau_array1,
                        #           fit_y0_array = fit_y0_array1,
                        #           sigma_y0_array = sigma_y0_array1
                        #           )
                    except: # no y0
                        print("Saving new, overwriting original filename ...")
                        popt2D[int(kx), int(ky)], pcov2D[int(kx), int(ky)] = popt_new, pcov_new  # replace values
                        np.savez(final_folder + f"/popt_pcov_2D_tol{tolerance}.npz",
                                 popt_2D=popt2D,
                                 pcov_2D=pcov2D)

                        fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[
                            kx, ky] = fitC0_new, fittau_new, sigmatau_new, sigmaC0_new
                        np.savez(final_folder + f"/fit_full_tol{tolerance}.npz",
                                 fit_C0_array=fit_C0_array1,
                                 fit_tau_array=fit_tau_array1,
                                 sigma_C0_array=sigma_C0_array1,
                                 sigma_tau_array=sigma_tau_array1
                                 )

                # continue to next graph:
                index += 1
                # clear graph2
                plt.clf()
                canvas2.draw()

    
# 1ST PART: FILE SELECTION DIALOG
# Create a root window (hidden)
root = tk.Tk()
root.title("Select wrong fits")
root.lift()

# Open the file explorer window for selecting multiple files
file_paths = filedialog.askopenfilenames()
root.destroy()

# Extract filenames from the file paths
file_names = [os.path.basename(file) for file in file_paths]
print(file_names)


# 2ND PART:
index = 0

root2 = tk.Tk()
root2.title("Fit corrections")

custom_style = ttk.Style()

# Set the font for all buttons and labels
custom_style.configure("TButton", font=("DejaVu Sans", 12))  
custom_style.configure("TLabel", font=("DejaVu Sans", 12)) 


fig=plt.figure(dpi=75)


def update_label_text():
    dynamic_label_var.set(f"Fit {index + 1} of {len(file_names)}")


dynamic_label_var = tk.StringVar()
dynamic_label = tk.Label(root2, textvariable=dynamic_label_var, font=("DejaVu Sans", 12))
dynamic_label.pack()



# Create the first canvas for old fit
label1 = tk.Label(root2, text="Old fit", font=("DejaVu Sans", 12))
label1.pack()

canvas1 = FigureCanvasTkAgg(fig, master=root2)
canvas1_widget = canvas1.get_tk_widget()
canvas1_widget.pack()

# Create the second canvas for new fit
label2 = tk.Label(root2, text="New fit", font=("DejaVu Sans", 12))
label2.pack()

canvas2 = FigureCanvasTkAgg(fig, master=root2)
canvas2_widget = canvas2.get_tk_widget()
canvas2_widget.pack()

DDM.set_root(root2, canvas1, canvas2)



# buttons and labels:
showold_button = tk.Button(root2, text="Show old fit", command=lambda:process(showold=True, refit=False, rewrite_and_continue=False), activebackground='SystemButtonFace')
# showold_button.pack()

entry_label = tk.Label(root2, text="new tau 2 upper bound:")
# entry_label.pack()

entry = tk.Entry(root2)
entry.insert(0,10000)
# entry.pack()

entry_label2 = tk.Label(root2, text="new tolerance:")
# entry_label.pack()

entry2 = tk.Entry(root2)
entry2.insert(0,tolerance)


entry_label3 = tk.Label(root2, text="new cutoff:")
# entry_label.pack()

entry3 = tk.Entry(root2)
entry3.insert(0,cutoff)


entry_label4 = tk.Label(root2, text="new tau2 lower bound:")
entry4 = tk.Entry(root2)
entry4.insert(0, 100)
# entry.pack()

refit_button = tk.Button(root2, text="Refit", command=lambda:process(showold=True, refit=True, rewrite_and_continue=False), activebackground='SystemButtonFace')
# refit_button.pack()

rewrite_button = tk.Button(root2, text="OVERWRITE and continue", command=lambda: [process(showold=True, refit=True, rewrite_and_continue=True), process(showold=True, refit=False, rewrite_and_continue=False, only_continue=False)], activebackground='SystemButtonFace')
# rewrite_button.pack()

skip_button = tk.Button(root2, text="SKIP", command=lambda: [process(showold=False, refit=False, rewrite_and_continue=False, only_continue=True), process(showold=True, refit=False, rewrite_and_continue=False, only_continue=False), canvas2.delete()], activebackground='SystemButtonFace')
# skip_button.pack()

# #test!
swaptau_button = tk.Button(root2, text="Use tau2 as tau1, OVERWRITE and continue", command=lambda: [process(showold=True, refit=True, rewrite_and_continue=True, swaptau=True), process(showold=True, refit=False, rewrite_and_continue=False, only_continue=False)], activebackground='SystemButtonFace')
# swaptau_button.pack()


entry_label.pack()
entry.pack()
entry_label2.pack()
entry2.pack()
entry_label3.pack()
entry3.pack()
entry_label4.pack()
entry4.pack()
padding = 1
showold_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=padding, expand=True)
refit_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=padding, expand=True)
rewrite_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=padding, expand=True)
skip_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=padding, expand=True)
swaptau_button.pack(side=tk.LEFT, fill=tk.BOTH, padx=padding,expand=True)

root2.mainloop()





















    
# for filename in file_names:

#     # Define the regular expression pattern to extract the information

#     pattern = r"corr_func_fit_(\w+)_(.*?)_mA_(.*?)_mT_kx(-?\d+)_ky(-?\d+)(_\d+)?.png" 
#     # Use re.search to find the pattern in the filename
#     match = re.search(pattern, filename)
    
#     if match:
#         sample = match.group(1)
#         current_str = match.group(2)
#         magnetic_field_str = match.group(3)
#         kx = int(match.group(4))
#         ky = int(match.group(5))    
#         print(f"Sample: {sample}")
#         print(f"Current (I): {current_str} mA")
#         print(f"Magnetic Field (B): {magnetic_field_str} mT")
#         print(f"kx: {kx}")
#         print(f"ky: {ky}")
#     else:
#         print("No match found in the filename.")
        
        
#     # step one: FIND THE CORRECT FOLDER
    
#     samplefolder = expfolder + f"/{sample}"
#     pattern2 = rf"(.+)_({current_str})_mA"
#     count = 0
#     for folders in os.listdir(samplefolder):
#         match = re.search(pattern2, folders)
#         if match:
#             count += 1
#             final_folder = samplefolder + f"/{match[0]}"
#             print(final_folder)
#         if count > 1:
#             print("More than one current match!")
    
#     # step two: LOAD FIT 
    
#     data = np.load(final_folder + "/" +  "corr.npz")
#     t, corr = data["t"] * deltat, data["corr"]
    
#     loaded_fit = np.load(final_folder + f"/fit_full_tol{tolerance}.npz")
#     fit_C0_array1, fit_tau_array1, sigma_C0_array1, sigma_tau_array1 = loaded_fit["fit_C0_array"], loaded_fit["fit_tau_array"], loaded_fit["sigma_C0_array"], loaded_fit["sigma_tau_array"]
#     fitC0, fittau, sigmatau, sigmaC0 = fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky]
    
#     datapc = np.load(final_folder + f"/popt_pcov_2D_tol{tolerance}.npz", allow_pickle=True)
#     popt2D, pcov2D = datapc["popt_2D"], datapc["pcov_2D"]
#     popt, pcov = datapc["popt_2D"][int(kx), int(ky)], datapc["pcov_2D"][int(kx), int(ky)]
    
#     # show old fit:
#     DDM.initial_settings(folder=final_folder, sample=sample, deltat_=deltat)
#     DDM.plot_fit_from_existing(corr, t, kx=kx, ky=ky, popt=popt, pcov=pcov, mag_field=float(magnetic_field_str), curr=float(current_str), plotshow=True)
    
#     # try new fit:
#     tau2upperbound = 1
#     fitC0_new, fittau_new, sigmatau_new, sigmaC0_new, fitC1_new, fittau2_new, popt_new, pcov_new = DDM.fit_corr(corr, t, kx, ky, bounds=([1, 0, 0, - np.inf], [3000, np.inf, np.inf, tau2upperbound]), showplot=True, old_return=False)
#     rewrite = False
    
#     # rewrite old fit:
#     if rewrite == True:
#         popt2D[int(kx), int(ky)], pcov2D[int(kx), int(ky)] = popt_new, pcov_new # replace values
#         np.savez(final_folder + f"/popt_pcov_2D_tol{tolerance}.npz",
#                 popt_2D=popt2D,
#                 pcov_2D=pcov2D)
        
#         fit_C0_array1[kx, ky], fit_tau_array1[kx, ky], sigma_tau_array1[kx, ky], sigma_C0_array1[kx, ky] = fitC0_new, fittau_new, sigmatau_new, sigmaC0_new
#         np.savez(final_folder + f"/fit_full_tol{tolerance}.npz",
#                   fit_C0_array = fit_C0_array1,
#                   fit_tau_array = fit_tau_array1,
#                   sigma_C0_array = sigma_C0_array1,
#                   sigma_tau_array = sigma_tau_array1             
#                   )

        
    #     # fit again in specific kx, ky point
        
    #     # show 
        
    #     # window? ... if OK, overwrite npz files...