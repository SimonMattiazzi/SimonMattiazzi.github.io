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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

### dtmm settings ###
from ML_DTMM_dtmm_settings import *

plt.rcParams.update({"figure.dpi":150,
                     "axes.grid":True,
                     "grid.color": "grey",
                     "grid.linestyle": ":",})

### Data preparation ###

# label generation settings:
NSTEPS_EACH_PAR = 6 # number of steps in each of the parameters range 
NVAR = 2 # number of slight Gaussian variations of input params for each parameter set
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

# 2. DATA: 

#PARAMS = np.asarray([INTENSITY,DFACTOR,TWIST,KSI])
# calculated:[1.29855351 0.87267592 0.19353709 5.5987824 ]
images_arr = np.empty((len(labels_arr), len(experiment_configurations), SHAPE[1], SHAPE[2], 3))
for i, label in tqdm(enumerate(labels_arr),total=len(labels_arr),ncols=100):
    # Simulate an array of images for every label:
    images = calculate_images(*label, exp_config=experiment_configurations)
    # images shape = (18, 11, 11, 3)
    images_arr[i] = images
      
# 3. Final data adjustments:

# # random permutation:
# # Shuffle data:
# permutation = np.random.permutation(len(labels_arr))
# learn_data_x = images_arr[permutation]
# learn_data_y = labels_arr[permutation]
    
# # first 90% = train
# x_train = learn_data_x [:int(len(learn_data_x)*0.9)]
# y_train = learn_data_y [:int(len(learn_data_x)*0.9)]

# # last 10% = test
# x_test = learn_data_x [int(len(learn_data_x)*0.9):]
# y_test = learn_data_y [int(len(learn_data_x)*0.9):]

#shuffling and splitting:
x_train, x_test, y_train, y_test = train_test_split(images_arr, labels_arr, shuffle=True, test_size=0.2, random_state=42)


# 4. CNN:

# Define the CNN model for regression of 4 parameters
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

# conv3D instead of conv2D for 

# Compile the model
cnn_model.compile(optimizer=Adam(),
                  loss=MeanSquaredError(),
                  metrics=["mean_absolute_error"])

# Train the model
history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

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

# Examples for one parameter
param_index = 0  # Index of the parameter to plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, param_index], y_pred[:, param_index], color='blue', label='Predicted vs. Actual')
plt.plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted  Intensity')
plt.legend()
plt.grid(True)
plt.show()

# Examples for one parameter
param_index = 1  # Index of the parameter to plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, param_index], y_pred[:, param_index], color='blue', label='Predicted vs. Actual')
plt.plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted  Dfactor')
plt.legend()
plt.grid(True)
plt.show()

# Examples for one parameter
param_index = 2  # Index of the parameter to plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, param_index], y_pred[:, param_index], color='blue', label='Predicted vs. Actual')
plt.plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted  Twist')
plt.legend()
plt.grid(True)
plt.show()

# Examples for one parameter
param_index = 3  # Index of the parameter to plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, param_index], y_pred[:, param_index], color='blue', label='Predicted vs. Actual')
plt.plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted  Ksi')
plt.legend()
plt.grid(True)
plt.show()


# feed experimental data to check results:
import glob


def pixelate(img, resolution):
    height, width, _ = img.shape
    
    # Calculate the block size
    block_height = height // resolution
    block_width = width // resolution
    
    # Resize the image to the desired resolution
    resized_img = resize(img, (block_height * resolution, block_width * resolution), preserve_range=False)
    new_img = np.zeros((resolution, resolution,3))

    # Iterate over each block
    for i in range(0, block_height * resolution, block_height):
        for j in range(0, block_width * resolution, block_width):
            # Calculate the average color of the block
            avg_color = np.mean(resized_img[i:i+block_height, j:j+block_width], axis=(0, 1))#.astype(np.uint8)
            # Set the pixels in the block to the average color
            new_img[int(i/block_height), int(j/block_height)] = avg_color
            #print(new_img[int(i/block_height), int(j/block_height)])
    #new_img = downscale_local_mean(img,(resolution,resolution))
    return new_img


fnames = sorted(glob.glob("dtmm_measurement_2/dtmm2*.JPG"))#[::3]
raws = [plt.imread(fname) for fname in sorted(fnames)]

I,J = 990-20,2710-20
DI,DJ = 610+40,610+40

images = []

for im in raws:
    im = im[I:I+DI,J:J+DJ]
    images.append(im)
    #plt.figure()
    #plt.imshow(im)
    
cimages = []

for i,im in enumerate(images):
    im = pixelate(im/255., resolution=16)
    cimages.append(im)
    
