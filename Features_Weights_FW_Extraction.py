# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 07:16:35 2022

@author: llea4
"""

# %%
print('######################################################################')
print('###### Building the Models basic constances base on parameters #######')
print('################ & Features Extrtaction model with ###################')
print('######################        VGG16      #############################')
print("######################################################################")
print('########################## IMPORTING LIBRARIES #######################')

import os
import keras
from keras import models
from keras import layers
from keras.applications.vgg16 import VGG16
import UtilsLSTM_AFEW_FEx

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Activation

from keras.optimizers import SGD

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam

from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import ConvLSTM2D

import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, BatchNormalization ,LSTM
import shutil
from keras.preprocessing.image import ImageDataGenerator

import cv2
from matplotlib import pyplot as plt
from keras import backend as Kb
from keras import optimizers
from sklearn.decomposition import PCA

print("######################################################################")
# %%
print("######################################################################")
print("#############  Model_1_5000 Parameters Arrays generation #############")
print("######################################################################")

myarray_Para_Mod_1_5000 = np.fromfile('vgg16_1_5000F.h5', dtype=float)
print(myarray_Para_Mod_1_5000)
print(myarray_Para_Mod_1_5000.shape)
ma_1_5000 = myarray_Para_Mod_1_5000.shape[0]
myarray_Para_Mod_1_5000 = np.reshape(myarray_Para_Mod_1_5000, (ma_1_5000, 1))
myarray_Para_Mod_1_5000_NOnan = myarray_Para_Mod_1_5000[np.logical_not(np.isnan(myarray_Para_Mod_1_5000))]
myarray_Para_Mod_1_5000_NOnan_NOzeros = myarray_Para_Mod_1_5000_NOnan[myarray_Para_Mod_1_5000_NOnan != 0]
myarray_Para_Mod_1_5000_NOnan_NOzeros = np.reshape(myarray_Para_Mod_1_5000_NOnan_NOzeros, (myarray_Para_Mod_1_5000_NOnan_NOzeros.shape[0], 1))
print(myarray_Para_Mod_1_5000_NOnan_NOzeros.shape)

model_Para_Mod_1_5000_Set_1 = myarray_Para_Mod_1_5000_NOnan_NOzeros[0:25088,]
model_Para_Mod_1_5000_Set_2 = myarray_Para_Mod_1_5000_NOnan_NOzeros[100000:100000+25088,]
model_Para_Mod_1_5000_Set_3 = myarray_Para_Mod_1_5000_NOnan_NOzeros[200000:200000+25088,]

print('Model_1_5000 1st parameter set shape is:', model_Para_Mod_1_5000_Set_1.shape)
print('Model_1_5000 2nd parameter set shape is:', model_Para_Mod_1_5000_Set_2.shape)
print('Model_1_5000 3ed parameter set shape is:', model_Para_Mod_1_5000_Set_3.shape)
print("######################################################################")
# %%
print("######################################################################")
print("###################  Model_1_5000 total sets values ##################")
print("######################################################################")
t_M1_5000_S1 = 0
Mod_1_5000_Set_1 = 0
for t_M1_5000_S1 in range(0, len(model_Para_Mod_1_5000_Set_1)):
    Mod_1_5000_Set_1 =  Mod_1_5000_Set_1 + model_Para_Mod_1_5000_Set_1[t_M1_5000_S1, 0]
    t_M1_5000_S1 = t_M1_5000_S1 + 1
print('the Model 1_5000 & Test Set 1 total is        :', Mod_1_5000_Set_1)

t_M1_5000_S2 = 0
Mod_1_5000_Set_2 = 0
for t_M1_5000_S2 in range(0, len(model_Para_Mod_1_5000_Set_2)):
    Mod_1_5000_Set_2 =  Mod_1_5000_Set_2 + model_Para_Mod_1_5000_Set_2[t_M1_5000_S2, 0]
    t_M1_5000_S2 = t_M1_5000_S2 + 1
print('the Model 1_5000 & Test Set 2 total is        :', Mod_1_5000_Set_2)

t_M1_5000_S3 = 0
Mod_1_5000_Set_3 = 0
for t_M1_5000_S3 in range(0, len(model_Para_Mod_1_5000_Set_3)):
    Mod_1_5000_Set_3 =  Mod_1_5000_Set_3 + model_Para_Mod_1_5000_Set_3[t_M1_5000_S3, 0]
    t_M1_5000_S3 = t_M1_5000_S3 + 1
print('the Model 1_5000 & Test Set 3 total is        :', Mod_1_5000_Set_3)

# %%
print("######################################################################")
print("#############  Model_3_4000 Parameters Arrays generation #############")
print("######################################################################")
import numpy as np 
myarray_Para_Mod_3_4000 = np.fromfile('vgg16_3_4000M.h5', dtype=float)
print(myarray_Para_Mod_3_4000)
print(myarray_Para_Mod_3_4000.shape)
ma_3_4000 = myarray_Para_Mod_3_4000.shape[0]
myarray_Para_Mod_3_4000 = np.reshape(myarray_Para_Mod_3_4000, (ma_3_4000, 1))
myarray_Para_Mod_3_4000_NOnan = myarray_Para_Mod_3_4000[np.logical_not(np.isnan(myarray_Para_Mod_3_4000))]
myarray_Para_Mod_3_4000_NOnan_NOzeros = myarray_Para_Mod_3_4000_NOnan[myarray_Para_Mod_3_4000_NOnan != 0]
myarray_Para_Mod_3_4000_NOnan_NOzeros = np.reshape(myarray_Para_Mod_3_4000_NOnan_NOzeros, (myarray_Para_Mod_3_4000_NOnan_NOzeros.shape[0], 1))
print(myarray_Para_Mod_3_4000_NOnan_NOzeros.shape)

model_Para_Mod_3_4000_Set_1 = myarray_Para_Mod_3_4000_NOnan_NOzeros[0:25088,]
model_Para_Mod_3_4000_Set_2 = myarray_Para_Mod_3_4000_NOnan_NOzeros[100000:100000+25088,]
model_Para_Mod_3_4000_Set_3 = myarray_Para_Mod_3_4000_NOnan_NOzeros[200000:200000+25088,]

print('Model_3_4000 1st parameter set shape is:', model_Para_Mod_3_4000_Set_1.shape)
print('Model_3_4000 2nd parameter set shape is:', model_Para_Mod_3_4000_Set_2.shape)
print('Model_3_4000 3ed parameter set shape is:', model_Para_Mod_3_4000_Set_3.shape)
print("######################################################################")
# %%
print("######################################################################")
print("###################  Model_3_4000 total sets values ##################")
print("######################################################################")
t_M3_4000_S1 = 0
Mod_3_4000_Set_1 = 0
for t_M3_4000_S1 in range(0, len(model_Para_Mod_3_4000_Set_1)):
    Mod_3_4000_Set_1 =  Mod_3_4000_Set_1 + model_Para_Mod_3_4000_Set_1[t_M3_4000_S1, 0]
    t_M3_4000_S1 = t_M3_4000_S1 + 1
print('the Model 3_4000 & Test Set 1 total is        :', Mod_3_4000_Set_1)

t_M3_4000_S2 = 0
Mod_3_4000_Set_2 = 0
for t_M3_4000_S2 in range(0, len(model_Para_Mod_3_4000_Set_2)):
    Mod_3_4000_Set_2 =  Mod_3_4000_Set_2 + model_Para_Mod_3_4000_Set_2[t_M3_4000_S2, 0]
    t_M3_4000_S2 = t_M3_4000_S2 + 1
print('the Model 3_4000 & Test Set 2 total is        :', Mod_3_4000_Set_2)

t_M3_4000_S3 = 0
Mod_3_4000_Set_3 = 0
for t_M3_4000_S3 in range(0, len(model_Para_Mod_3_4000_Set_3)):
    Mod_3_4000_Set_3 =  Mod_3_4000_Set_3 + model_Para_Mod_3_4000_Set_3[t_M3_4000_S3, 0]
    t_M3_4000_S3 = t_M3_4000_S3 + 1
print('the Model 3_4000 & Test Set 3 total is        :', Mod_3_4000_Set_3)

# %%
print("######################################################################")
print("#############  Model_4_650 Parameters Arrays generation ##############")
print("######################################################################")

import numpy as np 
myarray_Para_Mod_4_650 = np.fromfile('vgg16_4_650F.h5', dtype=float)
print(myarray_Para_Mod_4_650)
print(myarray_Para_Mod_4_650.shape)
ma_4_650 = myarray_Para_Mod_4_650.shape[0]
myarray_Para_Mod_4_650 = np.reshape(myarray_Para_Mod_4_650, (ma_4_650, 1))
myarray_Para_Mod_4_650_NOnan = myarray_Para_Mod_4_650[np.logical_not(np.isnan(myarray_Para_Mod_4_650))]
myarray_Para_Mod_4_650_NOnan_NOzeros = myarray_Para_Mod_4_650_NOnan[myarray_Para_Mod_4_650_NOnan != 0]
myarray_Para_Mod_4_650_NOnan_NOzeros = np.reshape(myarray_Para_Mod_4_650_NOnan_NOzeros, (myarray_Para_Mod_4_650_NOnan_NOzeros.shape[0], 1))
print(myarray_Para_Mod_4_650_NOnan_NOzeros.shape)

model_Para_Mod_4_650_Set_1 = myarray_Para_Mod_4_650_NOnan_NOzeros[0:25088,]
model_Para_Mod_4_650_Set_2 = myarray_Para_Mod_4_650_NOnan_NOzeros[100000:100000+25088,]
model_Para_Mod_4_650_Set_3 = myarray_Para_Mod_4_650_NOnan_NOzeros[200000:200000+25088,]

print('Model_4_650 1st parameter set shape is:', model_Para_Mod_4_650_Set_1.shape)
print('Model_4_650 2nd parameter set shape is:', model_Para_Mod_4_650_Set_2.shape)
print('Model_4_650 3ed parameter set shape is:', model_Para_Mod_4_650_Set_3.shape)
print("######################################################################")
# %%
print("######################################################################")
print("###################  Model_4_650 total sets values ###################")
print("######################################################################")
t_M4_650_S1 = 0
Mod_4_650_Set_1 = 0
for t_M4_650_S1 in range(0, len(model_Para_Mod_4_650_Set_1)):
    Mod_4_650_Set_1 =  Mod_4_650_Set_1 + model_Para_Mod_4_650_Set_1[t_M4_650_S1, 0]
    t_M4_650_S1 = t_M4_650_S1 + 1
print('the Model 4_T1 & Test Set 1 total is        :', Mod_4_650_Set_1)

t_M4_650_S2 = 0
Mod_4_650_Set_2 = 0
for t_M4_650_S2 in range(0, len(model_Para_Mod_4_650_Set_2)):
    Mod_4_650_Set_2 =  Mod_4_650_Set_2 + model_Para_Mod_4_650_Set_2[t_M4_650_S2, 0]
    t_M4_650_S2 = t_M4_650_S2 + 1
print('the Model 4_T1 & Test Set 2 total is        :', Mod_4_650_Set_2)

t_M4_650_S3 = 0
Mod_4_650_Set_3 = 0
for t_M4_650_S3 in range(0, len(model_Para_Mod_4_650_Set_3)):
    Mod_4_650_Set_3 =  Mod_4_650_Set_3 + model_Para_Mod_4_650_Set_3[t_M4_650_S3, 0]
    t_M4_650_S3 = t_M4_650_S3 + 1
print('the Model 4_T1 & Test Set 3 total is        :', Mod_4_650_Set_3)

# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("######################## Mod 1 / TESTSETS ############################")
print('######################################################################')
print('###################### building the VGG16 base #######################')
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base.summary()
print('########### building the VGG16 Features extraction model #############')
modelF = models.Sequential()
modelF.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF.add(conv_base)
modelF.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_1(directory, sample_count):
    features = np.zeros(shape=(sample_count, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count, 2))
    # Preprocess data
    train_generator = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                              data_augment=False,
                                                              batch_size=batch_size,
                                                              target_size=(img_width,
                                                                           img_height),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in train_generator:
        features_batch = modelF.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# %%
print("######################################################################")
print("##### T & V directories for the original dataset in all Models #######")
print("######################################################################")
print('################### features and labels generation ###################')
training_directory_Original_Mod_1_Testset = "dataset/mod_1/dataset_org_modified/train_testset_200"
training_directory_Original_Mod_1_Trainset = "dataset/mod_1/dataset_org_modified/train_trainset_200"

training_directory_Original_Mod_3_Testset = "dataset/mod_3/AFEW/Faces/training_testset_200"
training_directory_Original_Mod_3_Trainset = "dataset/mod_3/AFEW/Faces/training_trainset_200"

training_directory_Original_Mod_4_Testset = "dataset/mod_4/dataset/train_testset_200"
training_directory_Original_Mod_4_Trainset = "dataset/mod_4/dataset/train_trainset_200"

print("################ Training mod1 variables extracting ##################")
list_Testset_Original_Mod_1 = os.listdir(training_directory_Original_Mod_1_Testset)
classes_number_Testset_Original_Mod_1 = len(list_Testset_Original_Mod_1)

i = 0
final_Testset_Original_Mod_1 = 0
for i in range(0, len(list_Testset_Original_Mod_1)):
    training_directory_Testset_i = training_directory_Original_Mod_1_Testset + "/" + list_Testset_Original_Mod_1[i]
    list_Testset_i = os.listdir(training_directory_Testset_i)
    number_files_Testset_i = len(list_Testset_i)
    print(number_files_Testset_i, list_Testset_Original_Mod_1[i])
    final_Testset_Original_Mod_1 = number_files_Testset_i + final_Testset_Original_Mod_1
    print("total Testset samples:", final_Testset_Original_Mod_1)
    i = i + 1
print("######################################################################")

print("############## Training_Train/mod1 variables extracting ##############")
list_Trainset_Original_Mod_1 = os.listdir(training_directory_Original_Mod_1_Trainset)
classes_number_Trainset_Original_Mod_1 = len(list_Trainset_Original_Mod_1)

i = 0
final_Trainset_Original_Mod_1 = 0
for i in range(0, len(list_Trainset_Original_Mod_1)):
    training_directory_Trainset_i = training_directory_Original_Mod_1_Trainset + "/" + list_Trainset_Original_Mod_1[i]
    list_Trainset_i = os.listdir(training_directory_Trainset_i)
    number_files_Trainset_i = len(list_Trainset_i)
    print(number_files_Trainset_i, list_Trainset_Original_Mod_1[i])
    final_Trainset_Original_Mod_1 = number_files_Trainset_i + final_Trainset_Original_Mod_1
    print("total Trainset samples:", final_Trainset_Original_Mod_1)
    i = i + 1
print("######################################################################")

print("################## Training_mod3 variables extracting ################")
list_Testset_Original_Mod_3 = os.listdir(training_directory_Original_Mod_3_Testset)
classes_number_Testset_Original_Mod_3 = len(list_Testset_Original_Mod_3)

i = 0
final_Testset_Original_Mod_3 = 0
for i in range(0, len(list_Testset_Original_Mod_3)):
    training_directory_T13_i = training_directory_Original_Mod_3_Testset + "/" + list_Testset_Original_Mod_3[i]
    list_T13_i = os.listdir(training_directory_T13_i)
    number_files_T13_i = len(list_T13_i)
    print(number_files_T13_i, list_Testset_Original_Mod_3[i])
    final_Testset_Original_Mod_3 = number_files_T13_i + final_Testset_Original_Mod_3
    print("total Testset samples:", final_Testset_Original_Mod_3)
    i = i + 1
print("######################################################################")

print("############# Training_Train/mod3 variables extracting ###############")
list_Trainset_Original_Mod_3 = os.listdir(training_directory_Original_Mod_3_Trainset)
classes_number_Trainset_Original_Mod_3 = len(list_Trainset_Original_Mod_3)

i = 0
final_Trainset_Original_Mod_3 = 0
for i in range(0, len(list_Trainset_Original_Mod_3)):
    training_directory_Trainset_i = training_directory_Original_Mod_3_Trainset + "/" + list_Trainset_Original_Mod_3[i]
    list_Trainset_i = os.listdir(training_directory_Trainset_i)
    number_files_Trainset_i = len(list_Trainset_i)
    print(number_files_Trainset_i, list_Trainset_Original_Mod_3[i])
    final_Trainset_Original_Mod_3 = number_files_Trainset_i + final_Trainset_Original_Mod_3
    print("total Trainset samples:", final_Trainset_Original_Mod_3)
    i = i + 1
print("######################################################################")

print("################ Training_mod4 variables extracting ##################")
list_Testset_Original_Mod_4 = os.listdir(training_directory_Original_Mod_4_Testset)
classes_number_Testset_Original_Mod_4 = len(list_Testset_Original_Mod_4)

i = 0
final_Testset_Original_Mod_4 = 0
for i in range(0, len(list_Testset_Original_Mod_4)):
    training_directory4_Testset_i = training_directory_Original_Mod_4_Testset + "/" + list_Testset_Original_Mod_4[i]
    list4_Testset_i = os.listdir(training_directory4_Testset_i)
    number_files4_Testset_i = len(list4_Testset_i)
    print(number_files4_Testset_i, list_Testset_Original_Mod_4[i])
    final_Testset_Original_Mod_4 = number_files4_Testset_i + final_Testset_Original_Mod_4
    print("total Testset samples:", final_Testset_Original_Mod_4)
    i = i + 1
print("######################################################################")

print("############# Training_Train/mod4 variables extracting ###############")
list_Trainset_Original_Mod_4 = os.listdir(training_directory_Original_Mod_4_Trainset)
classes_number_Trainset_Original_Mod_4 = len(list_Trainset_Original_Mod_4)

i = 0
final_Trainset_Original_Mod_4 = 0
for i in range(0, len(list_Trainset_Original_Mod_4)):
    training_directory_Trainset_i = training_directory_Original_Mod_4_Trainset + "/" + list_Trainset_Original_Mod_4[i]
    list_Trainset_i = os.listdir(training_directory_Trainset_i)
    number_files_Trainset_i = len(list_Trainset_i)
    print(number_files_Trainset_i, list_Trainset_Original_Mod_4[i])
    final_Trainset_Original_Mod_4 = number_files_Trainset_i + final_Trainset_Original_Mod_4
    print("total Trainset samples:", final_Trainset_Original_Mod_4)
    i = i + 1
print("######################################################################")

print("######################################################################")
objectsPERclass_Testset_Original_Mod_1 = final_Testset_Original_Mod_1 / classes_number_Testset_Original_Mod_1
objectsPERclass_Trainset_Original_Mod_1 = final_Trainset_Original_Mod_1 / classes_number_Trainset_Original_Mod_1
objectsPERclass_Testset_Original_Mod_3 = final_Testset_Original_Mod_3 / classes_number_Testset_Original_Mod_3
objectsPERclass_Trainset_Original_Mod_3 = final_Trainset_Original_Mod_3 / classes_number_Trainset_Original_Mod_3
objectsPERclass_Testset_Original_Mod_4 = final_Testset_Original_Mod_4 / classes_number_Testset_Original_Mod_4
objectsPERclass_Trainset_Original_Mod_4 = final_Trainset_Original_Mod_4 / classes_number_Trainset_Original_Mod_4

classes_number_Testset_Original_Mod_1 = int(classes_number_Testset_Original_Mod_1)
classes_number_Trainset_Original_Mod_1 = int(classes_number_Trainset_Original_Mod_1)
classes_number_Testset_Original_Mod_3 = int(classes_number_Testset_Original_Mod_3)
classes_number_Trainset_Original_Mod_3 = int(classes_number_Trainset_Original_Mod_3)
classes_number_Testset_Original_Mod_4 = int(classes_number_Testset_Original_Mod_4)
classes_number_Trainset_Original_Mod_4 = int(classes_number_Trainset_Original_Mod_4)

objectsPERclass_Testset_Original_Mod_1 = int(objectsPERclass_Testset_Original_Mod_1)
objectsPERclass_Trainset_Original_Mod_1 = int(objectsPERclass_Trainset_Original_Mod_1)
objectsPERclass_Testset_Original_Mod_3 = int(objectsPERclass_Testset_Original_Mod_3)
objectsPERclass_Trainset_Original_Mod_3 = int(objectsPERclass_Trainset_Original_Mod_3)
objectsPERclass_Testset_Original_Mod_4 = int(objectsPERclass_Testset_Original_Mod_4)
objectsPERclass_Trainset_Original_Mod_4 = int(objectsPERclass_Trainset_Original_Mod_4)


print('the number of object in each 1/Testset  training class   is:', objectsPERclass_Testset_Original_Mod_1)
print('the number of object in each 1/Traintset  training class   is:', objectsPERclass_Trainset_Original_Mod_1)
print('the number of object in each 3/Testset  training class   is:', objectsPERclass_Testset_Original_Mod_3)
print('the number of object in each 3/Trainset  training class   is:', objectsPERclass_Trainset_Original_Mod_3)
print('the number of object in each 4/Testset  training class   is:', objectsPERclass_Testset_Original_Mod_4)
print('the number of object in each 4/Trainset  training class   is:', objectsPERclass_Trainset_Original_Mod_4)

print('the number of classes used for training_1/Testset  is:', classes_number_Testset_Original_Mod_1)
print('the number of classes used for training_1/Trainset  is:', classes_number_Trainset_Original_Mod_1)
print('the number of classes used for training_3/Testset  is:', classes_number_Testset_Original_Mod_3)
print('the number of classes used for training_3/Trainset  is:', classes_number_Trainset_Original_Mod_3)
print('the number of classes used for training_4/Testset is:', classes_number_Testset_Original_Mod_4)
print('the number of classes used for training_4/Trainset  is:', classes_number_Trainset_Original_Mod_4)

print("######################################################################")
# %%
print("######################################################################")
print("###### Extract features & reshapping VGG16 for Model 1_TESTSET #######")
print("######################################################################")
training_generator_features_VGG16_Original_Mod_1_Testset, training_labels_VGG16_Original_Mod_1_Testset = extract_features_1(training_directory_Original_Mod_1_Testset, final_Testset_Original_Mod_1)
print("training___generator_features_VGG16_Original_Mod_1 shape is:", training_generator_features_VGG16_Original_Mod_1_Testset.shape)
print("training___labels_VGG16_Original_Mod_1_Testset shape is:", training_labels_VGG16_Original_Mod_1_Testset.shape)

training_generator_features_VGG16_Original_Mod_1_Testset = np.reshape(training_generator_features_VGG16_Original_Mod_1_Testset, (final_Testset_Original_Mod_1, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_1_Testset shape is:", training_generator_features_VGG16_Original_Mod_1_Testset.shape)

print("######################################################################")
# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("######################## Mod 1 / TRAINSETS ###########################")
print('######################################################################')
print('###################### building the VGG16 base #######################')
conv_base_tr1 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base_tr1.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base_tr1.summary()
print('########### building the VGG16 Features extraction model #############')
modelF_tr1 = models.Sequential()
modelF_tr1.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF_tr1.add(conv_base_tr1)
modelF_tr1.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size_tr1 = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_1_tr1(directory, sample_count_tr1):
    features_tr1 = np.zeros(shape=(sample_count_tr1, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels_tr1 = np.zeros(shape=(sample_count_tr1, 2))
    # Preprocess data
    train_generator_tr1 = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                                  data_augment=False,
                                                                  batch_size=batch_size_tr1,
                                                                  target_size=(img_width,
                                                                               img_height),
                                                                  color_mode='grayscale',
                                                                  class_mode='categorical',
                                                                  shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch_tr1, labels_batch_tr1 in train_generator_tr1:
        features_batch_tr1 = modelF_tr1.predict(inputs_batch_tr1)
        features_tr1[i * batch_size_tr1: (i + 1) * batch_size_tr1] = features_batch_tr1
        labels_tr1[i * batch_size_tr1: (i + 1) * batch_size_tr1] = labels_batch_tr1
        i += 1
        if i * batch_size_tr1 >= sample_count_tr1:
            break
    return features_tr1, labels_tr1

print("######################################################################")
print("###### Extract features & reshapping VGG16 for Model 1_TRAINSET ######")
print("######################################################################")
training_generator_features_VGG16_Original_Mod_1_Trainset, training_labels_VGG16_Original_Mod_1_Trainset = extract_features_1_tr1(training_directory_Original_Mod_1_Trainset, final_Trainset_Original_Mod_1)
print("training___generator_features_VGG16_Original_Mod_1 shape is:", training_generator_features_VGG16_Original_Mod_1_Trainset.shape)
print("training___labels_VGG16_Original_Mod_1_Trainset shape is:", training_labels_VGG16_Original_Mod_1_Trainset.shape)

training_generator_features_VGG16_Original_Mod_1_Trainset = np.reshape(training_generator_features_VGG16_Original_Mod_1_Trainset, (final_Trainset_Original_Mod_1, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_1_Trainset shape is:", training_generator_features_VGG16_Original_Mod_1_Trainset.shape)

print("######################################################################")
# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("######################## Mod 3 / TESTSETS ############################")
print('######################################################################')
print('###################### building the VGG16 base #######################')
conv_base_3 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base_3.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base_3.summary()
print('########### building the VGG16 Features extraction model #############')
modelF_3 = models.Sequential()
modelF_3.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF_3.add(conv_base_3)
modelF_3.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size_3 = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_3(directory, sample_count_3):
    features_3 = np.zeros(shape=(sample_count_3, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels_3 = np.zeros(shape=(sample_count_3, 2))
    # Preprocess data
    train_generator_3 = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                              data_augment=False,
                                                              batch_size=batch_size_3,
                                                              target_size=(img_width,
                                                                           img_height),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch_3, labels_batch_3 in train_generator_3:
        features_batch_3 = modelF_3.predict(inputs_batch_3)
        features_3[i * batch_size_3: (i + 1) * batch_size_3] = features_batch_3
        labels_3[i * batch_size_3: (i + 1) * batch_size_3] = labels_batch_3
        i += 1
        if i * batch_size_3 >= sample_count_3:
            break
    return features_3, labels_3

print("######################################################################")
print("####### Extract features & reshapping VGG16 for Model 3_Testset ######")
print("######################################################################")
training_generator_features_VGG16_Original_Mod_3_Testset, training_labels_VGG16_Original_Mod_3_Testset = extract_features_3(training_directory_Original_Mod_3_Testset, final_Testset_Original_Mod_3)
print("training___generator_features_VGG16_Original_Mod_3 shape is:", training_generator_features_VGG16_Original_Mod_3_Testset.shape)
print("training___labels_VGG16_Original_Mod_3_Testset shape is:", training_labels_VGG16_Original_Mod_3_Testset.shape)

training_generator_features_VGG16_Original_Mod_3_Testset = np.reshape(training_generator_features_VGG16_Original_Mod_3_Testset, (final_Testset_Original_Mod_3, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_3_Testset shape is:", training_generator_features_VGG16_Original_Mod_3_Testset.shape)

print("######################################################################")
# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("######################## Mod 3 / TRAINSETS ###########################")
print('######################################################################')
print('###################### building the VGG16 base #######################')
conv_base_3_tr3 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base_3_tr3.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base_3_tr3.summary()
print('########### building the VGG16 Features extraction model #############')
modelF_3_tr3 = models.Sequential()
modelF_3_tr3.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF_3_tr3.add(conv_base_3_tr3)
# modelF.add(layers.Flatten())
modelF_3_tr3.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size_3_tr3 = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_3_tr3(directory, sample_count_3_tr3):
    features_3_tr3 = np.zeros(shape=(sample_count_3_tr3, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels_3_tr3 = np.zeros(shape=(sample_count_3_tr3, 2))
    # Preprocess data
    train_generator_3_tr3 = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                              data_augment=False,
                                                              batch_size=batch_size_3_tr3,
                                                              target_size=(img_width,
                                                                           img_height),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch_3_tr3, labels_batch_3_tr3 in train_generator_3_tr3:
        features_batch_3_tr3 = modelF_3_tr3.predict(inputs_batch_3_tr3)
        features_3_tr3[i * batch_size_3_tr3: (i + 1) * batch_size_3_tr3] = features_batch_3_tr3
        labels_3_tr3[i * batch_size_3_tr3: (i + 1) * batch_size_3_tr3] = labels_batch_3_tr3
        i += 1
        if i * batch_size_3_tr3 >= sample_count_3_tr3:
            break
    return features_3_tr3, labels_3_tr3

print("######################################################################")
print("###### Extract features & reshapping VGG16 for Model 3_Testset #######")
print("######################################################################")
print("")
print("")
print("")
training_generator_features_VGG16_Original_Mod_3_Trainset, training_labels_VGG16_Original_Mod_3_Trainset = extract_features_3_tr3(training_directory_Original_Mod_3_Trainset, final_Trainset_Original_Mod_3)
print("training___generator_features_VGG16_Original_Mod_3 shape is:", training_generator_features_VGG16_Original_Mod_3_Trainset.shape)
print("training___labels_VGG16_Original_Mod_3_Trainset shape is:", training_labels_VGG16_Original_Mod_3_Trainset.shape)

training_generator_features_VGG16_Original_Mod_3_Trainset = np.reshape(training_generator_features_VGG16_Original_Mod_3_Trainset, (final_Trainset_Original_Mod_3, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_3_Trainset shape is:", training_generator_features_VGG16_Original_Mod_3_Trainset.shape)

print("######################################################################")
# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("############################ Mod_4/Testset ###########################")
print("######################################################################")
print('###################### building the VGG16 base #######################')
conv_base_11 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base_11.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base_11.summary()
print('########### building the VGG16 Features extraction model #############')
modelF_11 = models.Sequential()
modelF_11.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF_11.add(conv_base_11)
modelF_11.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size_11 = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_11(directory, sample_count_11):
    features_11 = np.zeros(shape=(sample_count_11, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels_11 = np.zeros(shape=(sample_count_11, 2))
    # Preprocess data
    train_generator_11 = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                              data_augment=False,
                                                              batch_size=batch_size_11,
                                                              target_size=(img_width,
                                                                           img_height),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch_11, labels_batch_11 in train_generator_11:
        features_batch_11 = modelF_11.predict(inputs_batch_11)
        features_11[i * batch_size_11: (i + 1) * batch_size_11] = features_batch_11
        labels_11[i * batch_size_11: (i + 1) * batch_size_11] = labels_batch_11
        i += 1
        if i * batch_size_11 >= sample_count_11:
            break
    return features_11, labels_11

print("######################################################################")
print("###### Extract features & reshapping VGG16 for Model 4_TESTSET #######")
print("######################################################################")
training_generator_features_VGG16_Original_Mod_4_Testset, training_labels_VGG16_Original_Mod_4_Testset = extract_features_11(training_directory_Original_Mod_4_Testset, final_Testset_Original_Mod_4)
print("training___generator_features_VGG16_Original_Mod_4 shape is:", training_generator_features_VGG16_Original_Mod_4_Testset.shape)
print("training___labels_VGG16_Original_Mod_4_Testset shape is:", training_labels_VGG16_Original_Mod_4_Testset.shape)

training_generator_features_VGG16_Original_Mod_4_Testset = np.reshape(training_generator_features_VGG16_Original_Mod_4_Testset, (final_Testset_Original_Mod_4, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_4_Testset shape is:", training_generator_features_VGG16_Original_Mod_4_Testset.shape)

print("######################################################################")
# %%
print('######################################################################')
print('########### Building the Features Extrtaction model with #############')
print('############################### VGG16 ################################')
print("######################## Mod 4 / TRAINSETS ###########################")
print('######################################################################')
print('###################### building the VGG16 base #######################')
conv_base_4_tr4 = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
conv_base_4_tr4.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
conv_base_4_tr4.summary()
print('########### building the VGG16 Features extraction model #############')
modelF_4_tr4 = models.Sequential()
modelF_4_tr4.add(layers.Conv2D(3, kernel_size=(1, 1), input_shape=(224, 224, 1), activation='relu'))
modelF_4_tr4.add(conv_base_4_tr4)
modelF_4_tr4.summary()
print('############ Code the VGG16 Features extraction program ##############')
batch_size_4_tr4 = 1
img_width, img_height = 224, 224  # Default input size for VGG16
VGG16_Feature_Dim_1 = 7
VGG16_Feature_Dim_2 = 7
VGG16_Feature_Dim_3 = 512

def extract_features_4_tr4(directory, sample_count_4_tr4):
    features_4_tr4 = np.zeros(shape=(sample_count_4_tr4, VGG16_Feature_Dim_1, VGG16_Feature_Dim_2, VGG16_Feature_Dim_3))
    # Must be equal to the output of the convolutional base
    labels_4_tr4 = np.zeros(shape=(sample_count_4_tr4, 2))
    # Preprocess data
    train_generator_4_tr4 = UtilsLSTM_AFEW_FEx.image_data_generator(directory,
                                                              data_augment=False,
                                                              batch_size=batch_size_4_tr4,
                                                              target_size=(img_width,
                                                                           img_height),
                                                              color_mode='grayscale',
                                                              class_mode='categorical',
                                                              shuffle=True)

    # Pass data through convolutional base
    i = 0
    for inputs_batch_4_tr4, labels_batch_4_tr4 in train_generator_4_tr4:
        features_batch_4_tr4 = modelF_4_tr4.predict(inputs_batch_4_tr4)
        features_4_tr4[i * batch_size_4_tr4: (i + 1) * batch_size_4_tr4] = features_batch_4_tr4
        labels_4_tr4[i * batch_size_4_tr4: (i + 1) * batch_size_4_tr4] = labels_batch_4_tr4
        i += 1
        if i * batch_size_4_tr4 >= sample_count_4_tr4:
            break
    return features_4_tr4, labels_4_tr4
print("######################################################################")
print("###### Extract features & reshapping VGG16 for Model 3_Trainset ######")
print("######################################################################")
training_generator_features_VGG16_Original_Mod_4_Trainset, training_labels_VGG16_Original_Mod_4_Trainset = extract_features_4_tr4(training_directory_Original_Mod_4_Trainset, final_Trainset_Original_Mod_4)
print("training___generator_features_VGG16_Original_Mod_4 shape is:", training_generator_features_VGG16_Original_Mod_4_Trainset.shape)
print("training___labels_VGG16_Original_Mod_4_Trainset shape is:", training_labels_VGG16_Original_Mod_4_Trainset.shape)

training_generator_features_VGG16_Original_Mod_4_Trainset = np.reshape(training_generator_features_VGG16_Original_Mod_4_Trainset, (final_Trainset_Original_Mod_4, (VGG16_Feature_Dim_1*VGG16_Feature_Dim_2*VGG16_Feature_Dim_3)))
print("training___generator_features_VGG16_Original_Mod_4_Trainset shape is:", training_generator_features_VGG16_Original_Mod_4_Trainset.shape)

print("######################################################################")
# %%
print("######################################################################")
print("###############  Analysed Arrays for Model 1_Testset #################")
print("######################################################################")
Analysed_Training_Array_Original_Mod_1_Testset = ()
Analysed_Training_Array_Original_Mod_1_Testset = training_generator_features_VGG16_Original_Mod_1_Testset

print("Analysed_Training___Array_Original_Mod_1_Testset shape is:", Analysed_Training_Array_Original_Mod_1_Testset.shape)

print("######################################################################")
print("####################  Analysed Arrays for Model 1_Trainset #################")
print("######################################################################")
Analysed_Training_Array_Original_Mod_1_Trainset = ()
Analysed_Training_Array_Original_Mod_1_Trainset = training_generator_features_VGG16_Original_Mod_1_Trainset

print("Analysed_Training___Array_Original_Mod_1_Trainset shape is:", Analysed_Training_Array_Original_Mod_1_Trainset.shape)

print("######################################################################")
print("##################  Analysed Arrays for Model 3_Testset ##############")
print("######################################################################")
Analysed_Training_Array_Original_Mod_3_Testset = ()
Analysed_Training_Array_Original_Mod_3_Testset = training_generator_features_VGG16_Original_Mod_3_Testset

print("Analysed_Training___Array_Original_Mod_3_Testset shape is:", Analysed_Training_Array_Original_Mod_3_Testset.shape)

print("######################################################################")
print("##############  Analysed Arrays for Model 3_Trainset #################")
print("######################################################################")
Analysed_Training_Array_Original_Mod_3_Trainset = ()
Analysed_Training_Array_Original_Mod_3_Trainset = training_generator_features_VGG16_Original_Mod_3_Trainset

print("Analysed_Training___Array_Original_Mod_3_Trainset shape is:", Analysed_Training_Array_Original_Mod_3_Trainset.shape)

print("######################################################################")
print("###############  Analysed Arrays for Model 4_Testset #################")
print("######################################################################")
Analysed_Training_Array_Original_Mod_4_Testset = ()
Analysed_Training_Array_Original_Mod_4_Testset = training_generator_features_VGG16_Original_Mod_4_Testset

print("Analysed_Training___Array_Original_Mod_4_Testset shape is:", Analysed_Training_Array_Original_Mod_4_Testset.shape)

print("######################################################################")
print("################  Analysed Arrays for Model 4_Trainset ###############")
print("######################################################################")
Analysed_Training_Array_Original_Mod_4_Trainset = ()
Analysed_Training_Array_Original_Mod_4_Trainset = training_generator_features_VGG16_Original_Mod_4_Trainset

print("Analysed_Training___Array_Original_Mod_4_Trainset shape is:", Analysed_Training_Array_Original_Mod_4_Trainset.shape)
