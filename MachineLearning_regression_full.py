# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:26:24 2024

This is the first attempt to build a regression neural network
that would be able to determine the sizes and concentrations
of nanoparticles in a solution based on the characteristic
dependence of autocorrelation function amplitude on the
scattering vector as measured with c-DDM.

First, a Pandas dataframe with limited experimental data is loaded,
then, a lot of additional data is crated artificially, based on distribution
of experimental amplitudes for each combination of size and concentration.
(to be replaced with a numerical simulation once it is working properly)

This data is then used for training the regressional neural network.

At the end, some quick analysis is presented.
Overall, the results seem good.

@author: Simon Mattiazzi
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from IPython import get_ipython
from matplotlib import cm
from matplotlib.colors import Normalize

get_ipython().run_line_magic('matplotlib', 'inline') # inline plots

plt.rcParams.update({"figure.dpi":150,
                     "axes.grid":True,
                     "grid.color": "grey",
                     "grid.linestyle": ":",})


SAVEFOLDER= "D:\\Users Data\\Simon\\MachineLearning\\saved_data\\"
FILENAME = "test2"


# pandas dataframe settings:
#----------------------------------------------------------------------------#
# actual NA values:
NA_dict = {"NA_1": 0.098,
           "NA_2": 0.146,
           "NA_3": 0.195,
           "NA_4": 0.293,
           "NA_5": 0.366}

# actual concentration values in [mg (PS spheres) /ml (water)]:
conc_dict_mg_ml = {"1_1": 0.980,
                   "1_2": 0.490,
                   "1_4": 0.245,
                   "1_8": 0.123,
                   "1_16": 0.061,
                   "1_32": 0.031}

# Load dataframe:
filepath = "D:\\Users Data\\Simon\\Electrophoresis experiments\\Full_results_library\\full_library_raw.pkl"
full_results_library = pd.read_pickle(filepath)
k = np.load("D:\\Users Data\\Simon\\Electrophoresis experiments\\Test_concentrations\\196_nm\\NA_1\\1_1\\run_0\\k.npy")[3:]

#----------------------------------------------------------------------------#


# Function definitions:

def get_entry(dataframe, size, concentration, NA):
    """
    Function to read an entry from the key.
    """
    key = (size, concentration, NA)
    if key in dataframe.index:
        return dataframe.loc[key]["temperature"], dataframe.loc[key]["amplitudes_array"]
    else:
        return [],[]
        #return None


def interpolate_general_qs(general_q, standard_q):
    """
    Input must be in such a form that x component follows standard q array = np.arange(128)*kstep: 
    """
    # TO DO
    return 0


def interpolate_nans(array):
    """
    Interpolates NaN values in a given numpy array using linear interpolation.
    """
    # Convert to a pandas Series
    series = pd.Series(array)
    
    # Interpolate the NaN values
    interpolated_series = series.interpolate()
    
    # Convert back to a numpy array
    interpolated_array = interpolated_series.to_numpy()
    
    return interpolated_array


def create_training_data_single(size_nm, conc_num, NA_num, N_rep, cutoff=None, plot=False):
    """
    Generates training data from measured data by creating several similar curves by adding appropriate noise.
    Data is normalized, but amplitude data is preserved by filling half (because of weight) of each array with constant log(amplitude)/30 value.

    Parameters:
    -----------
    size_nm : float
        Size parameter in nanometers.
    conc_num : float
        Concentration parameter in numerical value.
    NA_num : float
        Numerical Aperture parameter.
    N_rep : int
        Number of repetitions for generating modified arrays with noise.
    cutoff : int, optional
        Number of elements to retain in each amplitude array. If None, all elements are retained. Default is None.
    plot : bool, optional
        If True, plots the original, modified, and full arrays. Default is False.

    Returns:
    --------
    out_arr_mag : numpy.ndarray
        Used for x_train data. A 2D numpy array containing the modified amplitude data with magnitude information.
    labels_arr : numpy.ndarray
        Used for y_train data. A 1D numpy array with constant (size_nm, conc_num) values, corresponding to the length of out_arr_mag.
    """
    norm = 1# NA_num**2 * conc_N_ml(conc_num, size_nm)
    

    temp, amplitudes = get_entry(dataframe=full_results_library,
                        size=size_nm,
                        concentration=conc_num,
                        NA=NA_num)
    
    if len(amplitudes) == 0: #empty ones
        print("Empty: ", size_nm, conc_num, NA_num)
        return None

    
    if cutoff != None:
        amplitudes=amplitudes[:,:cutoff]

    out_arr = np.empty((0, amplitudes.shape[1]))
    
    # Get st. dev
    yerr=np.nanstd(amplitudes, axis=0) / norm

    # Interpolate NaN values and store to out
    amplitudes = np.array([interpolate_nans(arr) for arr in amplitudes])
    out_arr = np.append(out_arr, amplitudes, axis=0)
         
    # Generate N_rep modified arrays with normal distributed variations
    modified_arrays = np.array([
        savgol_filter(
        np.random.normal(array_run, yerr / 1, (N_rep, array_run.size))
        , 50 ,10)
        for array_run in amplitudes
    ])
    
    # Reshape modified_arrays and append to out_arr
    modified_arrays = modified_arrays.reshape(-1, amplitudes.shape[1])
    out_arr = np.append(out_arr, modified_arrays, axis=0)
    
    # Add magnitude information:
    magnitudes = np.array([np.max(array) for array in out_arr])
    out_arr_mag = np.array([np.concatenate((
        np.ones(len(out_arr[0])) * np.log(magnitudes[i])/30, arr/magnitudes[i]))
        for i, arr in enumerate(out_arr)])
    
    if plot:
        plt.figure()
        for amplitude in amplitudes: # original ones
            plt.plot(amplitude, c="black", linestyle="--" ,linewidth=2, zorder=10)
    
        for modified_array in modified_arrays: # modified ones
            plt.plot(modified_array) 
        
        for arr in out_arr.reshape(((N_rep + 1) * len(amplitudes)), len(amplitudes[0])): # full
            plt.plot(arr)
        plt.show()
    
    labels_arr = np.concatenate((np.ones((len(out_arr),1)) * size_nm, np.ones((len(out_arr),1)) * conc_num), axis=1)
    
    return(out_arr_mag, labels_arr)


# Create training data:
def create_training_data_full(nrep):
    learn_data_x, learn_data_y = None, None # will be initialized with the first appropriate array
    
    for size_nm in [196, 295, 508]:   
        for conc_str in ["1_1", "1_2", "1_4", "1_8", "1_16", "1_32"]:
            
            # initialize array with the first working element, dimensions currently unknown:
            if learn_data_x is None: 
                training_data = create_training_data_single(size_nm, conc_dict_mg_ml[conc_str], NA_dict["NA_5"], N_rep=NREP, cutoff=80, plot=False)
                if training_data != None:
                    learn_data_x, learn_data_y = training_data
                    
            # append all others:
            else:
                training_data = create_training_data_single(size_nm, conc_dict_mg_ml[conc_str], NA_dict["NA_5"], N_rep=NREP, cutoff=80, plot=False)
                if training_data != None:
                    learn_data_i_x, learn_data_i_y = training_data
                    
                    learn_data_x = np.append(learn_data_x, learn_data_i_x, axis=0)
                    learn_data_y = np.append(learn_data_y, learn_data_i_y, axis=0)
    
    # Shuffle data:
    permutation = np.random.permutation(len(learn_data_x))
    learn_data_x = learn_data_x[permutation]
    learn_data_y = learn_data_y[permutation]
                
    #save:
    np.save(f"{SAVEFOLDER}reg_learn_data_x{FILENAME}.npy", learn_data_x)                
    np.save(f"{SAVEFOLDER}reg_learn_data_y{FILENAME}.npy", learn_data_y)     
    
    return learn_data_x, learn_data_y


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.5


# Build a neural network (this architecture seems to work best):
def build_model():
    model = Sequential([
        Input(shape=(len(x_train[0]),)),
        Dense(512, activation='leaky_relu'),
        Dense(256, activation='leaky_relu'),
        Dense(128, activation='leaky_relu'),
        Dense(64, activation='leaky_relu'),
        Dense(2, activation="linear") # because 2 params
    ])
    
    # Compile the model:
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model


#----------------------------------------------------------------------------#
# CREATE TRAINING AND TEST DATA:
    
NREP = 100000 # number of repetitions in creating data
   
# run:
learn_data_x, learn_data_y = create_training_data_full(nrep=NREP)

# load:
#learn_data_x = np.load(f"{SAVEFOLDER}reg_learn_data_x{FILENAME}.npy")
#learn_data_y = np.load(f"{SAVEFOLDER}reg_learn_data_y{FILENAME}.npy")

# first 90% = train
x_train = learn_data_x [:int(len(learn_data_x)*0.9)]
y_train = learn_data_y [:int(len(learn_data_x)*0.9)]

# last 10% = test
x_test = learn_data_x [int(len(learn_data_x)*0.9):]
y_test = learn_data_y [int(len(learn_data_x)*0.9):]

# Checks
print("nans in train", np.isnan(x_train).sum())  # Check for NaNs
print("infs in train", np.isinf(x_train).sum())  # Check for infinities
print("nans in labels", np.isnan(y_train).sum())  # Check for NaNs in labels
print("infs in labels", np.isinf(y_train).sum())  # Check for infinities in labels
print("class balance", np.unique(y_train, return_counts=True))  # Check for class balance

# Define additional neural network parameters:
lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# BUILD AND TRAIN THE NEURAL NETWORK
#------------------------------------------------------------------------#

# build a model:
model = build_model()

# Train the model:
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping, lr_scheduler])

# Save the entire model as a `.keras` zip archive.
model.save(f'{SAVEFOLDER}reg_model{FILENAME}.keras')
np.save(f"{SAVEFOLDER}reg_history_{FILENAME}.npy", history)

# # Load a fresh Keras model from the .keras zip archive:
#model = tf.keras.models.load_model(f'{SAVEFOLDER}reg_model{FILENAME}.keras')
#history = np.load(f"{SAVEFOLDER}reg_history_{FILENAME}.npy", allow_pickle=True).item()


# CHECKS
#---------------------------------------------------------------------------#

# Evaluate the model:
loss, mae = model.evaluate(x_test, y_test)
print(f'Test mae: {mae}')

# Summary to visualize the model
# model.summary()

# Plot training & validation Mean Absolute Error values:
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model Mean Absolute Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


np.set_printoptions(suppress=True, precision=2) # numpy number print without scientific format

# Quick visual test:
plt.figure()
for i in range(10):
    print(model.predict(x_test[i].reshape(1, -1))[0])
    prediction = model.predict(x_test[i].reshape(1, -1))[0] # 2D because batch predictions
    plt.plot(x_test[i], label="predicted " + str(prediction) + ", label " + str(y_test[i]))
plt.legend()

# Quick test: random point "in-between" (when trained on mixtures as well)
# plt.figure()
# for i in range(3):
#     prediction = model.predict((x_test[i]/2+x_test[i+1]/2).reshape(1, -1))[0] # 2D because batch predictions
#     plt.plot(x_test[i]/2+x_test[i+1]/2, label="predicted " + str(prediction) + ", label " + str(y_test[i]) + " + "+str(y_test[i+1]))
# plt.title("mixture")
# plt.legend()




# Predict on test data
y_pred = model.predict(x_test)


# all parameters pred + label:
for i in range(5):
    print("Predicted: ", y_pred[i])
    print("Label: ", y_test[i])


# HISTOGRAMS RELATIVE
fig, axs = plt.subplots(1, 2)

axs[0].hist((y_pred[:,0]-y_test[:,0])/y_test[:,0], bins=30)
axs[0].set_title("size")
axs[1].hist((y_pred[:,1]-y_test[:,1])/y_test[:,1], bins=30)
axs[1].set_title("conc")
plt.show(fig)

# HISTOGRAMS ABSOLUTE
fig, axs = plt.subplots(1, 2)
axs[0].hist(y_pred[:,0], bins=100)
axs[0].set_title("size")
axs[1].hist(y_pred[:,1], bins=100)
axs[1].set_title("conc")

plt.show(fig)


# EACH PARAMETER SEPARATELY:

fig, axs = plt.subplots(2, 1)

norm = Normalize(vmin=0, vmax=len(y_pred)-1)
cmap = cm.get_cmap('viridis')
colors = cmap(norm(range(len(y_pred))))
slicing = 50

param_index = 0  # Index of the parameter to plot
axs[0].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index], c=colors[::slicing], label='Predicted / Actual')
axs[0].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[0].set_xlabel('Actual')
axs[0].set_ylabel('Predicted')
axs[0].set_title("size")

param_index = 1  # Index of the parameter to plot
axs[1].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index], c=colors[::slicing], label='Predicted / Actual')
axs[1].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[1].set_xlabel('Actual')
axs[1].set_ylabel('Predicted')
axs[1].set_title("conc")

plt.suptitle(f'Actual vs. Predicted  params, showing every {slicing}th element')
plt.legend()
plt.show(fig)


# KORELACIJE NAPAK:
range1 = [[-0.1, 0.1],[-0.1, 0.1]]
for i1 in [0,1]:
    for i2 in [0,1]:
        if i1 != i2 and i1 < i2:
            plt.figure()
            plt.title(f"Correlations relative errors params {i1} and {i2}" )
            plt.hist2d((y_pred[:,i1]-y_test[:,i1])/y_test[:,i1], (y_pred[:,i2]-y_test[:,i2])/y_test[:,i2], bins=[10, 10], range=range1, cmap='grey')
            plt.xlabel(f"err {i1}")
            plt.ylabel(f"err {i2}")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()






# to do: don't include one concentration in training data, but include it in testing data.










# ---------------------GRID SEARCH-------------------------------------------

# def build_model(optimizer='adam'):
#     model = Sequential([
#         Input(shape=(len(x_train[0]),)),
#         Dense(512, activation='relu'),
#         Dense(256, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(64, activation='relu'),
#         Dense(1, activation="linear")
#     ])
#     model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
#     return model

# model = KerasRegressor(build_fn=build_model, verbose=1)
# param_grid = {'batch_size': [32, 64, 128], 'epochs': [20, 50, 100], 'optimizer': ['adam', 'rmsprop']}
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_result = grid.fit(x_train, y_train)

# print(f"Best: using {grid_result.best_params_}")
# Best: nan using {'batch_size': 32, 'epochs': 20, 'optimizer': 'adam'}

#----------------------------------------------------------------#
