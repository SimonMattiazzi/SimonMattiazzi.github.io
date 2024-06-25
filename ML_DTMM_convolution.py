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
Corresponding label: array of unknown material params, such as MOD, TWIST, XI, AMP ...

This approach is good because it reduces the number of unnecessary params.

Trying with convolutional NN architecture.

Data is simulated using DTMM modlue.

Written by Simon.
"""

import numpy as np
#import dtmm
import itertools
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

### dtmm settings ###
from ML_DTMM_dtmm_settings import * # specific separate file for all dtmm simulations etc




### Data preparation ###

# label generation settings:
NSTEPS_EACH_PAR = 20 # number of steps in each of the parameters range 
NVAR = 5 # number of slight Gaussian variations of input params for each parameter set
SIGMA_PERCENTAGE = 0.001
N_PARAMS = 4

# data generation (experiment) settings:
polarizers_arr = [90]
analizers_arr = [90, 120, 150, 180, 210, 240]
retarders_arr = ["none", "lambda/4", "lambda"]
experiment_configurations = list(itertools.product(polarizers_arr, analizers_arr, retarders_arr))

# 1. LABELS:
 
# Prepare arrays of parameter sets
mod_min, mod_max = 1 / 100, 1 / 10
twist_min, twist_max = np.pi / 180 * 0, np.pi / 180 * 30
ksi_min, ksi_max = 1, 11
amp_min, amp_max = np.pi / 180 * 0, np.pi / 180 * 359

mod_arr = np.arange(mod_min, mod_max, (mod_max-mod_min)/NSTEPS_EACH_PAR) # modulation constant
twist_arr = np.arange(twist_min, twist_max, (twist_max-twist_min)/NSTEPS_EACH_PAR) # max mid-plane twist
ksi_arr = np.arange(ksi_min, ksi_max, (ksi_max-ksi_min)/NSTEPS_EACH_PAR) # the ksi constant - see :funf:`ampl`
amp_arr = np.arange(amp_min, amp_max, (amp_max-amp_min)/NSTEPS_EACH_PAR) # surface alignment angle 

# Create all parameter combinations:
combinations = itertools.product(mod_arr, twist_arr, ksi_arr, amp_arr)
labels_arr = np.empty((NSTEPS_EACH_PAR**N_PARAMS * NVAR, N_PARAMS)) * 0

# Iterate through all parameter combinations and create their normal
# distributed variations:
i=-1
for mod, twist, ksi, amp in combinations:
    for n in range(NVAR): # number of variations:
        i += 1
        # pick from a normal distribution:
        mod_i = np.random.normal(mod, (mod_max - mod_min) * SIGMA_PERCENTAGE)
        twist_i = np.random.normal(twist, (twist_max - twist_min) * SIGMA_PERCENTAGE)
        ksi_i = np.random.normal(ksi, (ksi_max - ksi_min) * SIGMA_PERCENTAGE)
        amp_i = np.random.normal(amp, (amp_max - amp_min) * SIGMA_PERCENTAGE)
        # save to array of labels:
        labels_arr[i] = np.array([mod_i, twist_i, ksi_i, amp_i])


# 2. DATA: 

# kaj pa mod in amp?
#PARAMS = np.asarray([INTENSITY,DFACTOR,TWIST,KSI])
images_arr=[]
for label in labels_arr[:2]:
    params = np.asarray([1.3, 0.9, label[1], label[2]])
    # Simulate an array of images for every label:
    images = calculate_images(*params, exp_config=experiment_configurations)
    # shape (18, 11, 11, 3)
    # v settings spremeni v np
    images_arr.append(images)
      


# 3. Final data adjustments:

# random permutation:
# Shuffle data:
permutation = np.random.permutation(len(labels_arr))
learn_data_x = images_arr[permutation]
learn_data_y = labels_arr[permutation]
    
# first 90% = train
x_train = learn_data_x [:int(len(learn_data_x)*0.9)]
y_train = learn_data_y [:int(len(learn_data_x)*0.9)]

# last 10% = test
x_test = learn_data_x [int(len(learn_data_x)*0.9):]
y_test = learn_data_y [int(len(learn_data_x)*0.9):]

 #raje: npr.
     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. CNN:

# Define the CNN model for regression of 4 parameters
cnn_model = Sequential([
    InputLayer(input_shape=(18, 11, 11, 3)),  # Input shape: 18 images of size 11x11 with 3 channels
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4)  # Output layer for 4 parameters (no activation for regression)
])

# Compile the model
cnn_model.compile(optimizer=Adam(),
                  loss=MeanSquaredError())

# Train the model
cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))



############# testing ##################

# Predict on test data
y_pred = cnn_model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Example for one parameter
param_index = 0  # Index of the parameter to plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test[:, param_index], y_pred[:, param_index], color='blue', label='Predicted vs. Actual')
plt.plot([y_test[:, param_index].min(), y_test[:, param_index].max()], 
         [y_test[:, param_index].min(), y_test[:, param_index].max()], 
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs. Predicted Parameter {param_index + 1}')
plt.legend()
plt.grid(True)
plt.show()
