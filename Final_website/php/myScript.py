import requests
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from keras import layers
import tensorflow
from keras.layers import Flatten, Dense, Input, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.preprocessing import sequence, image
from keras import layers
from keras.optimizers import Adam
from keras.models import load_model
import os
from keras.callbacks import Callback
from keras.models import Sequential
import keras
from keras_efficientnets import EfficientNetB0
from keras_efficientnets import EfficientNetB5

file123 = open('mytxt.txt','r')

my_line = file123.readline()
file123.close()

x = my_line.split("+")
img_path= x[0]
user = x[1]

img = cv2.imread(img_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(96,96))
img = img/255.
img = np.reshape(img,[1,96,96,3])



vgg_model = load_model("ml_models/vgg16_2_model.hdf5")
vgg_prob=vgg_model.predict(img)



################################################################################################################################

effb0_model = EfficientNetB0(include_top=False, weights='ml_models/efficientnet-b0_notop.h5',pooling='avg',input_shape=(96,96,3))
x = effb0_model.output

x = Dense(32)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(8)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)

x = Dense(4)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(4)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


predictions = Dense(1, activation='sigmoid')(x)

effb0_model= Model(input = effb0_model.input, output = predictions)

adam = keras.optimizers.Adam(lr=0.0001)

effb0_model.compile(
          optimizer=adam,
          loss='binary_crossentropy',
          metrics=['binary_accuracy']
          )

effb0_model.load_weights('ml_models/efficient0_2_model.h5')


eff_prob=effb0_model.predict(img)
#####################################################################################################################################

#####################################################################################################################################
from keras_efficientnets import EfficientNetB5
effb5_model = EfficientNetB5(include_top=False, weights='ml_models/efficientnet-b5_notop.h5',pooling='avg',input_shape=(96,96,3))
x = effb5_model.output


x = Dense(32)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(16)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)


x = Dense(8)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)

x = Dense(4)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)

x = Dense(4)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)          
x = Dropout(0.25)(x)

predictions = Dense(1, activation='sigmoid')(x)

effb5_model= Model(input = effb5_model.input, output = predictions)


filepath="efficient7_1_model.hdf5" # checkpoint
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_binary_accuracy',save_weights_only=True, verbose=1, save_best_only=True, mode='max')
adam = keras.optimizers.Adam(lr=0.0001)


effb5_model.compile(
          optimizer=adam,
          loss='binary_crossentropy',
          metrics=['binary_accuracy'])


effb5_model.load_weights('ml_models/efficient7_1_model.hdf5')


effb5_prob=effb5_model.predict(img)


avg_result=( (vgg_prob[0][0]) + (eff_prob[0][0]) + (effb5_prob[0][0])*3 ) / 3 


file_xyz = open('myreport.txt','w')
#diagnosis=(str(final_classes[0]) + "+" + str(user))
if (avg_result < 0.5):
    file_xyz.write("Negative")
else:
    file_xyz.write("Positive")        
file_xyz.close()

import pyAesCrypt
# encryption/decryption buffer size - 64K
bufferSize = 64 * 1024
password = "123456"
# encrypt
pyAesCrypt.encryptFile(img_path, (img_path + ".aes" ) , password, bufferSize)

import os
os.remove((img_path))
