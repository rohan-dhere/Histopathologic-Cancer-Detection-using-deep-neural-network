#!/usr/bin/env python
#keep this line
import requests

file123 = open('mytxt.txt','r')

my_line = file123.readline()
file123.close()

x = my_line.split("+")
img_path= x[0]
user = x[1]

#print(img_path)
#print (user)

import keras
import cv2
import numpy as np

model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights ='mobilenet_1_0_224_tf.h5', input_tensor=None, pooling=None, classes=1000)

img = cv2.imread(img_path)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(224,224))
img = np.reshape(img,[1,224,224,3])

img = img/255.
classes=model.predict(img)
final_classes = classes.argmax(axis=-1)
#print (final_classes[0])

file_xyz = open('myreport.txt','w')
#diagnosis=(str(final_classes[0]) + "+" + str(user))
diagnosis=str(final_classes[0])
file_xyz.write(diagnosis)
file_xyz.close()

import pyAesCrypt
# encryption/decryption buffer size - 64K
bufferSize = 64 * 1024
password = "123456"
# encrypt
pyAesCrypt.encryptFile(img_path, (img_path + ".aes" ) , password, bufferSize)

import os
os.remove((img_path))