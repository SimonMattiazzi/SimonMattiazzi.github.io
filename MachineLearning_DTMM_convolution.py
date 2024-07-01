# -*- coding: utf-8 -*-
"""
First attempt to make a NN solve the inverse DTMM problem,
instead of the minimization of difference
between experiment and numerical model.

Building block of one input element = pixelated image whose colors and patterns are
determined by alignement within sample (unknown params) and the
configuration of polarizers and lambda plates around the microscope (known params).

Whole input element: array of such images in a predefined order
(e.g.: 0 (P0, A90, lambda/4), 1 (P0, A45, labda/4), ...).
Corresponding label: array of unknown material params, such as INTENSITY, DFACTOR, TWIST, XI ...

This approach is good because it reduces the number of unnecessary params.

Trying with convolutional neural network architecture.

Data is simulated using DTMM modlue. Constants like MOD and AMP must be set in DTMM settings file

Written by Simon.
"""

import numpy as np
#import dtmm
import itertools
import matplotlib.pyplot as plt
import time
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
import datetime

# timestamp:
timestamp = str(datetime.datetime.now()).replace(":", ".")[:-10]

### dtmm settings ###
from ML_DTMM_dtmm_settings import *

plt.rcParams.update({"figure.dpi":150,
                     "axes.grid":True,
                     "grid.color": "grey",
                     "grid.linestyle": ":",})


### Data preparation ###


FILENAME = "test1"
SAVEFOLDER= "D:\\Users Data\\Simon\\MachineLearning\\saved_data\\"

# label generation settings:
NSTEPS_EACH_PAR = 10 # number of steps in each of the parameters range 
NVAR = 4 # number of slight Gaussian variations of input params for each parameter set
SIGMA_PERCENTAGE = 0.001
N_PARAMS = 4

# data generation (experiment) settings:
polarizers_arr = [90]
analizers_arr = [90, 120, 150, 180, 210, 240]
retarders_arr = ["none", "lambda/4", "lambda"]
experiment_configurations = list(itertools.product(polarizers_arr, analizers_arr, retarders_arr))

# 1. LABELS:

# Prepare arrays of parameter sets:
intensity_min, intensity_max = 0.7,1.5
dfactor_min, dfactor_max = 0.5, 1
twist_min, twist_max = np.pi / 180 * 0, np.pi / 180 * 30
ksi_min, ksi_max = 1, 11

intensity_arr = np.linspace(intensity_min, intensity_max, NSTEPS_EACH_PAR)
dfactor_arr = np.linspace(dfactor_min, dfactor_max, NSTEPS_EACH_PAR) 
twist_arr = np.linspace(twist_min, twist_max, NSTEPS_EACH_PAR) # max mid-plane twist
ksi_arr = np.linspace(ksi_min, ksi_max, NSTEPS_EACH_PAR) # the ksi constant - see :funf:`ampl`

# Create all parameter combinations:
    
    
def generate_labels():
    combinations = itertools.product(intensity_arr, dfactor_arr, twist_arr, ksi_arr)
    labels_arr = np.empty((NSTEPS_EACH_PAR**N_PARAMS * NVAR, N_PARAMS)) * 0
    
    # Iterate through all parameter combinations and create their normal
    # distributed variations:
    i=-1
    for intensity, dfactor, twist, ksi in combinations:
        for n in range(NVAR): # number of variations:
            i += 1
            # pick from a normal distribution:
            intensity_i = np.random.normal(intensity, (intensity_max - intensity_min) * SIGMA_PERCENTAGE)
            dfactor_i = np.random.normal(dfactor, (dfactor_max - dfactor_min) * SIGMA_PERCENTAGE)
            twist_i = np.random.normal(twist, (twist_max - twist_min) * SIGMA_PERCENTAGE)
            ksi_i = np.random.normal(ksi, (ksi_max - ksi_min) * SIGMA_PERCENTAGE)
            # save to array of labels:
            labels_arr[i] = np.array([intensity_i, dfactor_i, twist_i, ksi_i])
    np.save(f"{SAVEFOLDER}labels_arr{FILENAME}.npy", labels_arr)
    return labels_arr
#PARAMS = np.asarray([INTENSITY,DFACTOR,TWIST,KSI])


def generate_data():
    images_arr = np.empty((len(labels_arr), len(experiment_configurations), SHAPE[1], SHAPE[2], 3))
    for i, label in tqdm(enumerate(labels_arr),total=len(labels_arr),ncols=100):
        # Simulate an array of images for every label:
        images = calculate_images(*label, exp_config=experiment_configurations)
        # images shape = (18, 11, 11, 3)
        images_arr[i] = images
    np.save(f"{SAVEFOLDER}images_arr{FILENAME}.npy", images_arr)
    return images_arr
        


# Define the CNN model for regression of 4 parameters
def build_model():
    cnn_model = Sequential([
        InputLayer(shape=(len(experiment_configurations), SHAPE[1], SHAPE[2], 3)),  # Input shape: 18 images of size 16x16 with 3 channels
        TimeDistributed(Conv2D(16, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4)  # Output layer for 4 parameters (no activation for regression)
    ])
    
    # Compile the model
    cnn_model.compile(optimizer=Adam(),
                      loss=MeanSquaredError(),
                      metrics=["mean_absolute_error"])
    
    return cnn_model


#############################################################


# OPTION A: BUILD FROM SCRATCH:
labels_arr = generate_labels()
images_arr = generate_data()
x_train, x_test, y_train, y_test = train_test_split(images_arr, labels_arr, shuffle=True, test_size=0.2, random_state=42)

cnn_model = build_model()

# Train the model:
history = cnn_model.fit(x_train, y_train, epochs=40, batch_size=32, validation_data=(x_test, y_test))
np.save(f"{SAVEFOLDER}history_{FILENAME}.npy", history)

# Save the entire model as a `.keras` zip archive.
cnn_model.save(f'{SAVEFOLDER}cnn_model_{FILENAME}.keras')
    


# OPTION B: LOAD
# labels_arr = np.load(f"{SAVEFOLDER}labels_arr{FILENAME}.npy")
# images_arr = np.load(f"{SAVEFOLDER}images_arr{FILENAME}.npy")
# x_train, x_test, y_train, y_test = train_test_split(images_arr, labels_arr, shuffle=True, test_size=0.2, random_state=42)

# # Reload a fresh Keras model from the .keras zip archive:
# cnn_model = tf.keras.models.load_model(f'cnn_model_{FILENAME}.keras')
# history = np.load(f"history_{FILENAME}.npy", allow_pickle=True).item()







############# testing ##################







# Plot training & validation Mean Absolute Error values:
plt.figure()
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


# Predict on test data
y_pred = cnn_model.predict(x_test)
# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")


# all 4 parameters pred + label:
for i in range(5):
    print("Predicted: ", y_pred[i])
    print("Label: ", y_test[i])



# HISTOGRAMS RELATIVE
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(y_pred[:,0]-y_test[:,0], bins=30)
axs[0, 0].set_title("Intensity")
axs[0, 1].hist(y_pred[:,1]-y_test[:,1], bins=30)
axs[0, 1].set_title("Dfactor")
axs[1, 0].hist(y_pred[:,2]-y_test[:,2], bins=30)
axs[1, 0].set_title("Twist")
axs[1, 1].hist(y_pred[:,3]-y_test[:,3], bins=30)
axs[1, 1].set_title("Ksi")
plt.show(fig)

# HISTOGRAMS ABSOLUTE
fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(y_pred[:,0], bins=100)
axs[0, 0].set_title("Intensity")
axs[0, 1].hist(y_pred[:,1], bins=100)
axs[0, 1].set_title("Dfactor")
axs[1, 0].hist(y_pred[:,2], bins=100)
axs[1, 0].set_title("Twist")
axs[1, 1].hist(y_pred[:,3], bins=100)
axs[1, 1].set_title("Ksi")
plt.show(fig)


# EACH PARAMETER SEPARATELY:

fig, axs = plt.subplots(2, 2)

norm = Normalize(vmin=0, vmax=len(y_pred)-1)
cmap = cm.get_cmap('viridis')
colors = cmap(norm(range(len(y_pred))))
slicing = 500

param_index = 0  # Index of the parameter to plot
axs[0,0].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index], c=colors[::slicing], label='Predicted / Actual')
axs[0,0].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[0,0].set_xlabel('Actual')
axs[0,0].set_ylabel('Predicted')
axs[0,0].set_title("Intensity")

param_index = 1  # Index of the parameter to plot
axs[0,1].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index], c=colors[::slicing], label='Predicted / Actual')
axs[0,1].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[0,1].set_xlabel('Actual')
axs[0,1].set_ylabel('Predicted')
axs[0,1].set_title("Dfactor")

param_index = 2  # Index of the parameter to plot
axs[1,0].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index], c=colors[::slicing],label='Predicted / Actual')
axs[1,0].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[1,0].set_xlabel('Actual')
axs[1,0].set_ylabel('Predicted')
axs[1,0].set_title("Twist")

param_index = 3  # Index of the parameter to plot
axs[1,1].scatter(y_test[::slicing, param_index], y_pred[::slicing, param_index],c=colors[::slicing], label='Predicted / Actual')
axs[1,1].plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axs[1,1].set_xlabel('Actual')
axs[1,1].set_ylabel('Predicted')
axs[1,1].set_title("Ksi")


plt.suptitle(f'Actual vs. Predicted  params, showing every {slicing}th element')
plt.legend()
plt.show(fig)



# KORELACIJE NAPAK:
range1 = [[-0.1, 0.1],[-0.1, 0.1]]
for i1 in [0,1,2,3]:
    for i2 in [0,1,2,3]:
        if i1 != i2 and i1 < i2:
            plt.figure()
            plt.title(f"Correlations relative errors params {i1} and {i2}" )
            plt.hist2d((y_pred[:,i1]-y_test[:,i1])/y_test[:,i1], (y_pred[:,i2]-y_test[:,i2])/y_test[:,i2], bins=[10, 10], range=range1, cmap='grey')
            plt.xlabel(f"err {i1}")
            plt.ylabel(f"err {i2}")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()







# # feed experimental data to check results:

# def pixelate(img, resolution):
#     height, width, _ = img.shape
    
#     # Calculate the block size
#     block_height = height // resolution
#     block_width = width // resolution
    
#     # Resize the image to the desired resolution
#     resized_img = resize(img, (block_height * resolution, block_width * resolution), preserve_range=False)
#     new_img = np.zeros((resolution, resolution,3))

#     # Iterate over each block
#     for i in range(0, block_height * resolution, block_height):
#         for j in range(0, block_width * resolution, block_width):
#             # Calculate the average color of the block
#             avg_color = np.mean(resized_img[i:i+block_height, j:j+block_width], axis=(0, 1))#.astype(np.uint8)
#             # Set the pixels in the block to the average color
#             new_img[int(i/block_height), int(j/block_height)] = avg_color
#             #print(new_img[int(i/block_height), int(j/block_height)])
#     #new_img = downscale_local_mean(img,(resolution,resolution))
#     return new_img


# fnames = sorted(glob.glob("dtmm_measurement_2/dtmm2*.JPG"))#[::3]
# raws = [plt.imread(fname) for fname in sorted(fnames)]

# I,J = 990-20,2710-20
# DI,DJ = 610+40,610+40

# images = []

# for im in raws:
#     im = im[I:I+DI,J:J+DJ]
#     images.append(im)
#     #plt.figure()
#     #plt.imshow(im)
    
# cimages = []

# for i,im in enumerate(images):
#     im = pixelate(im/255., resolution=16)
#     cimages.append(im)
    

# experimental_pred = cnn_model.predict(cimages)
