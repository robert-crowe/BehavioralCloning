from __future__ import print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, noise
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import optimizers
from keras.regularizers import l1, l2, l1l2
from keras.callbacks import EarlyStopping

import numpy as np
from sklearn.model_selection import train_test_split
import csv
from PIL import Image, ImageEnhance, ImageOps
import cv2
from img_process import process_img, img_config

# Indexes into driving_log
CENTER = 0
LEFT = 1
RIGHT = 2
STEERING = 3
THROTTLE = 4
BRAKE = 5
SPEED = 6

# driving_log contains absolute path to images as well as steering, throttle, etc.
with open('driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    X_train = list(reader)

X_train = np.array(X_train)

y_train = []
for row in X_train:
    y_train.append(row[STEERING])

y_train = np.array(y_train)

# Split between train, test, and validation sets:
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

print('img_config[resized_2Dshape] {}'.format(img_config['resized_2Dshape']))
print('img_config[resized_3Dshape] {}'.format(img_config['resized_3Dshape']))

# Generator
def generate_from_driving_log(X_data, y_data, batch_size, dataset_name):
    n_data = len(X_data)
    perm = np.random.permutation(len(X_data))
    X_data, y_data = X_data[perm], y_data[perm]
    while 1:
        for batch_num in range(n_data // batch_size + 1):
            X_list, y_list = [], []
            for b_idx in range(batch_size):
                x_idx = (batch_num * batch_size) + b_idx
                img = Image.open(X_data[x_idx % n_data][CENTER])
                img = process_img(img)
                img = np.asarray(img)
                X_list.append(img)
                y_list.append(y_data[x_idx % n_data])
            yield np.asarray(X_list), np.asarray(y_list)

model = Sequential()

init = 'he_normal'
reg = l2(l=0.0012)
print('Normal')

model.add(BatchNormalization(epsilon=0.001, mode=0, axis=3, momentum=0.99,
    beta_init=init, gamma_init=init, gamma_regularizer='l1', beta_regularizer='l1',
    input_shape=img_config['resized_3Dshape']))

print('Conv')
model.add(Convolution2D(25, 5, 5, W_regularizer=reg,
    init=init, activation='relu', border_mode='valid', subsample=(2, 2), bias=True))

print('Conv')
model.add(Convolution2D(36, 5, 5, W_regularizer=reg,
    init=init, activation='relu', border_mode='valid', subsample=(2, 2), bias=True))

print('Conv')
model.add(Convolution2D(48, 5, 5, W_regularizer=reg,
    init=init, activation='relu', border_mode='valid', subsample=(2, 2), bias=True))

print('Conv')
model.add(Convolution2D(64, 3, 3, W_regularizer=reg,
    init=init, activation='relu', border_mode='valid', subsample=(1, 1), bias=True))

print('Conv')
model.add(Convolution2D(64, 3, 3, W_regularizer=reg,
    init=init, activation='relu', border_mode='valid', subsample=(1, 1), bias=True))

print('Flat')
model.add(Flatten())

print('Dense')
model.add(Dense(100, init=init, activation='relu', W_regularizer=reg, bias=True))

print('Drop')
model.add(Dropout(p=0.5))

print('Dense')
model.add(Dense(50, init=init, activation='relu', W_regularizer=reg, bias=True))

print('Dense')
model.add(Dense(10, init=init, activation='relu', W_regularizer=reg, bias=True))

print('Dense')
model.add(Dense(1, init=init, activation='linear', W_regularizer=reg, bias=True))

opt = optimizers.Adamax(lr=0.0001)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])


batch_size = 64
n_epoch = 10000
val_batch_size = 32

# Fit the model on the batches generated from driving log
print('Begin training')
earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=21, verbose=1, mode='min')
model.fit_generator(generate_from_driving_log(X_train, y_train, batch_size, 'Training'),
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=n_epoch,
                    validation_data=generate_from_driving_log(X_val, y_val, val_batch_size, 'Validation'),
                    nb_val_samples=val_batch_size, callbacks=[earlyStop]
                    )

# Testing
print('Generating test data')
def process_testset(X_data, y_data):
    X_list, y_list = [], []
    for idx in range(len(X_data)):
        img = Image.open(X_data[idx][CENTER])
        img = process_img(img)
        img = np.asarray(img)
        X_list.append(img)
        y_list.append(y_data[idx])
    return np.asarray(X_list), np.asarray(y_list)

X_test, y_test = process_testset(X_test, y_test)

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss: {}'.format(score[0]))
print('Test MSE: {}'.format(score[1]))

# Save model and weights
file = open("model.json", "w")
file.write(model.to_json())
file.close()

model.save_weights('model.h5')