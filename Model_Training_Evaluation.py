# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 05:27:20 2022

@author: llea4
"""

# %%
print('(1)-(1)-(1)------------------------------------------------(1)-(1)-(1)')
print('######################################################################')
print('###### Building the Models basic constances base on parameters #######')
print('################ & Features Extrtaction model with ###################')
print('######################        VGG16      #############################')
print("######################################################################")
print('########################## IMPORTING LIBRARIES #######################')
from keras import layers
from keras import models
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
print("######################################################################")
# %%
print('######################################################################')
print('##########  Build the Image_Data_Generator Function  #################')
print("######################################################################")
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def image_data_generator(data_dir, data_augment=False, batch_size=50, #change batch size according to step size
                         target_size=(224, 224), color_mode='rgb',  # grayscale',
                         class_mode='categorical', shuffle=True):
    if data_augment:
        datagen = ImageDataGenerator(rescale=1./255, rotation_range=1,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2, shear_range=0.2,
                                     zoom_range=0.2, horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(data_dir, target_size=target_size,
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            class_mode=class_mode)
    return generator

print("######################################################################")
# %%
print('######################################################################')
print('######### Building the Models basic Features Extrtaction  ############')
print('###################### model with VGG16  #############################')
print("######################################################################")
print('######################### for Mod_1 ##################################')
print("######################################################################")

model_1 = models.Sequential()
model_1.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_1.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_1.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_1.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_1.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model_1.add(layers.Flatten())
model_1.add(layers.Dense(units=4096,activation="relu"))
model_1.add(layers.Dense(units=4096,activation="relu"))
model_1.add(layers.Dense(units=2, activation="softmax"))

model_1.summary()

# %%
print('######################################################################')
print('######################  Compile Model 1  #############################')
print("######################################################################")
from keras import optimizers

model_1.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6), metrics=['acc'])

print("######################################################################")
# %%
print('######################################################################')
print('#########################  Training Model 1 ##########################')
print("######################################################################")
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'dataset')

print('######################### for Mod_1 ##################################')
mod_1 = os.path.join(dataset_dir, 'mod_1/dataset_org_modified')
snips_1 = os.path.join(base_dir, 'snips_F/mod_1')

train_dir_Mod1_5000 = os.path.join(mod_1, 'train_5000')
validation_dir_Mod1_1000 = os.path.join(mod_1, 'val_1000')
test_dir_Mod1_200 = os.path.join(mod_1, 'train_testset_200')
train_dir_Mod1_200 = os.path.join(mod_1, 'train_trainset_200')

train_generator_Mod1_5000 = image_data_generator(train_dir_Mod1_5000)
validation_generator_Mod1_1000 = image_data_generator(validation_dir_Mod1_1000)
# %%
import pickle
import pandas as pd
print("###################  Training _Model 1  ##############################")
checkpoint_1_5000 = ModelCheckpoint("vgg16_1_5000F_40.h5", monitor='val_acc', verbose=1,
                                    save_best_only=True, save_weights_only=False,
                                    mode='auto', period=1)
early_1_5000 = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
history_1_5000 = model_1.fit_generator(train_generator_Mod1_5000, steps_per_epoch=250,
                                       epochs=30, validation_data=validation_generator_Mod1_1000,
                                       validation_steps=100, callbacks=[checkpoint_1_5000,early_1_5000])

# Save training history as a history dictionary file so we can reload late to draw the accuracy and loss curves
Figures_dir_M1 = os.path.join(base_dir, 'Figures/Mod_1')
path_figures_M1 = os.path.join(Figures_dir_M1, 'Mod_1_30_')
with open(path_figures_M1 + 'histroyDict', 'wb') as file_pi_M1:
    pickle.dump(history_1_5000.history, file_pi_M1)

# Save training history as a CSV file
# Convert the history.history dict to a pandas DataFrame:    
hist_df = pd.DataFrame(history_1_5000.history) 
hist_df.to_csv(path_figures_M1 + 'history.csv')

print("###################  End Training _Model 1  ##########################")
# %%
# Model 3 4000
print('######################################################################')
print('######### Building the Models basic Features Extrtaction  ############')
print('###################### model with VGG16  #############################')
print("######################################################################")
print('######################### for Mod_3 ##################################')
print("######################################################################")

model_3 = models.Sequential()
model_3.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_3.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_3.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_3.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_3.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model_3.add(layers.Flatten())
model_3.add(layers.Dense(units=4096,activation="relu"))
model_3.add(layers.Dense(units=4096,activation="relu"))
model_3.add(layers.Dense(units=2, activation="softmax"))

model_3.summary()

print('######################################################################')
print('#######################  Compile Model 3  ############################')
print("######################################################################")
from keras import optimizers

model_3.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6), metrics=['acc'])

print('######################################################################')
print('##########################  Training Model 3 #########################')
print("######################################################################")
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
import pandas as pd


base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'dataset')

print('######################### for Mod_3 ##################################')
mod_3 = os.path.join(dataset_dir, 'mod_3/AFEW/Faces')
snips_3 = os.path.join(base_dir, 'snips_F/mod_3')

train_dir_Mod3_4000 = os.path.join(mod_3, 'train_4000')
validation_dir_Mod3_1000 = os.path.join(mod_3, 'val_1000')
test_dir_Mod3_200 = os.path.join(mod_3, 'training_testset_200')
train_dir_Mod3_200 = os.path.join(mod_3, 'training_trainset_200')


train_generator_Mod3_4000 = image_data_generator(train_dir_Mod3_4000)
validation_generator_Mod3_1000 = image_data_generator(validation_dir_Mod3_1000)

print("################### Training _Model 3 ################################")
checkpoint_3_4000 = ModelCheckpoint("vgg16_3_4000F.h5", monitor='val_acc', verbose=1,
                                  save_best_only=True, save_weights_only=False,
                                  mode='auto', period=1)
early_3_4000 = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
history_3_4000 = model_3.fit_generator(train_generator_Mod3_4000, steps_per_epoch=400,
                                     epochs=20, validation_data=validation_generator_Mod3_1000,
                                     validation_steps=100, callbacks=[checkpoint_3_4000,early_3_4000])

Figures_dir_M3 = os.path.join(base_dir, 'Figures/Mod_3')
path_figures_M3 = os.path.join(Figures_dir_M3, 'Mod_3')
with open(path_figures_M3 + 'histroyDict', 'wb') as file_pi:
    pickle.dump(history_3_4000.history, file_pi)

# Save training history as a CSV file
# Convert the history.history dict to a pandas DataFrame:    
hist_df_M3 = pd.DataFrame(history_3_4000.history) 
hist_df_M3.to_csv(path_figures_M3 + 'history.csv')


print("###################  End Training _Model 3  ##########################")
# %%
# Model 4 650
print('######################################################################')
print('######### Building the Models basic Features Extrtaction  ############')
print('###################### model with VGG16  #############################')
print("######################################################################")
print('######################### for Mod_4 ##################################')
print("######################################################################")

model_4 = models.Sequential()
model_4.add(layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_4.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_4.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_4.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_4.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

model_4.add(layers.Flatten())
model_4.add(layers.Dense(units=4096,activation="relu"))
model_4.add(layers.Dense(units=4096,activation="relu"))
model_4.add(layers.Dense(units=2, activation="softmax"))

model_4.summary()

print('######################################################################')
print('##########################  Compile Model 4  #########################')
print("######################################################################")

from keras import optimizers

model_4.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6), metrics=['acc'])

print("######################################################################")

print('######################################################################')
print('##########################  Training Model 4 #########################')
print("######################################################################")
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'dataset')

print('######################### for Mod_4 ##################################')
mod_4 = os.path.join(dataset_dir, 'mod_4/dataset')
snips_4 = os.path.join(base_dir, 'snips_F/mod_4')

train_dir_Mod4_650 = os.path.join(mod_4, 'train_650')
validation_dir_Mod4_300 = os.path.join(mod_4, 'test_300')
test_dir_Mod4_200 = os.path.join(mod_4, 'train_testset_200')
train_dir_Mod4_200 = os.path.join(mod_4, 'train_trainset_200')

train_generator_Mod4_650 = image_data_generator(train_dir_Mod4_650)
validation_generator_Mod4_300 = image_data_generator(validation_dir_Mod4_300)

print("################### Training _Model 4 ################################")
checkpoint_4_650 = ModelCheckpoint("vgg16_4_650F_50.h5", monitor='val_acc', verbose=1,
                                  save_best_only=True, save_weights_only=False,
                                  mode='auto', period=1)
early_4_650 = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
history_4_650 = model_4.fit_generator(train_generator_Mod4_650, steps_per_epoch=27,
                                     epochs=30, validation_data=validation_generator_Mod4_300,
                                     validation_steps=12, callbacks=[checkpoint_4_650,early_4_650])

Figures_dir_M4 = os.path.join(base_dir, 'Figures/Mod_4')
path_figures_M4 = os.path.join(Figures_dir_M4, 'Mod_4')
with open(path_figures_M4 + 'histroyDict', 'wb') as file_pi:
    pickle.dump(history_4_650.history, file_pi)

# Save training history as a CSV file
# Convert the history.history dict to a pandas DataFrame:    
hist_df_M4 = pd.DataFrame(history_4_650.history) 
hist_df_M4.to_csv(path_figures_M4 + 'history_2030.csv')

print("###################  End Training _Model 4  ##########################")
# %%
print('######################################################################')
print('##### Plot the Prediction plots for Model_1, Model_3, & Model_4 ######')
print("######################################################################")
from matplotlib import pyplot as plt

print('######################### for Mod_1 ##################################')
results = model_1.predict(train_generator_Mod1_5000)
results1 = model_1.predict(validation_generator_Mod1_1000)
plt.scatter(range(10000),results[:, 1], c='r')
plt.scatter(range(2000),results1[:, 1], c='g')
plt.title("model_1 Prediction")
plt.show()

plt.plot(history_1_5000.history['loss'])
plt.title("Model_1 Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(history_1_5000.history['acc'])
plt.title("Model_1 Accuracy")
plt.ylabel("Acuracy")
plt.xlabel("Epoch")
plt.show()

import matplotlib.pyplot as plt
plt.plot(history_1_5000.history["acc"])
plt.plot(history_1_5000.history['val_acc'])
plt.plot(history_1_5000.history['loss'])
plt.plot(history_1_5000.history['val_loss'])
plt.title("Model_1 Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

print('######################### for Mod_3 ##################################')
results3 = model_3.predict(train_generator_Mod3_4000)
results31 = model_3.predict(validation_generator_Mod3_1000)
plt.scatter(range(8000),results3[:, 1], c='r')
plt.scatter(range(2000),results31[:, 1], c='g')
plt.title("Model_3 Prediction")
plt.show()

plt.plot(history_3_4000.history['loss'])
plt.title("Model_3 Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(history_3_4000.history['acc'])
plt.title("Model_3 Accuracy")
plt.ylabel("Acuracy")
plt.xlabel("Epoch")
plt.show()

import matplotlib.pyplot as plt
plt.plot(history_3_4000.history["acc"])
plt.plot(history_3_4000.history['val_acc'])
plt.plot(history_3_4000.history['loss'])
plt.plot(history_3_4000.history['val_loss'])
plt.title("Model_3 Accuracy & Loss")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

print('######################### for Mod_4 ##################################')
results4 = model_4.predict(train_generator_Mod4_650)
results41 = model_4.predict(validation_generator_Mod4_300)
plt.scatter(range(1300),results4[:, 1], c='r')
plt.scatter(range(600),results41[:, 1], c='g')
plt.title("Model_4 Prediction")
plt.show()

plt.plot(history_4_650.history['loss'])
plt.title("Model_4 Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

plt.plot(history_4_650.history['acc'])
plt.title("Model_4 Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

import matplotlib.pyplot as plt
plt.plot(history_4_650.history["acc"])
plt.plot(history_4_650.history['val_acc'])
plt.plot(history_4_650.history['loss'])
plt.plot(history_4_650.history['val_loss'])
plt.title("Model_4 Accuracy & Loss")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

print('######################################################################')
# %%
print('######################################################################')
print('########### plot the loss & accuracy for model 1, 3, & 4  ############')
print("######################################################################")
from matplotlib import pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plt_acc_loss(history, acc_title, loss_title, acc_filepath=None,
                 loss_filepath=None):
    # plt the accuracy and the loss picture in `history`, and save
    # the figure to png directory with title name
    if not acc_filepath:
        acc_filepath = acc_title
    if not loss_filepath:
        loss_filepath = loss_title
    acc_filepath = os.path.join(snips, acc_filepath)
    loss_filepath = os.path.join(snips, loss_filepath)
    
    acc = smooth_curve(history.history['acc'])
    val_acc = smooth_curve(history.history['val_acc'])
    loss = smooth_curve(history.history['loss'])
    val_loss = smooth_curve(history.history['val_loss'])
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title(acc_title)
    plt.legend()
    plt.grid(True)
    plt.savefig(acc_filepath)
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(loss_title)
    plt.grid(True)
    plt.legend()
    plt.savefig(loss_filepath)
    plt.show()

print('######################### for Mod_1 ##################################')
snips = snips_1
plt_acc_loss(history_1_5000, 'cnn_accuracy_Mod_1_rgb', 'cnn_loss_Mod_1', 'cnn_accuracy(95.86%).png', 'cnn_loss(0.1863).png')

print('######################### for Mod_3 ##################################')
snips = snips_3
plt_acc_loss(history_3_4000, 'cnn_accuracy_Mod_3', 'cnn_loss_Mod_3_rgb', 'cnn_accuracy(99.8%).png', 'cnn_loss(0.322).png')

print('######################### for Mod_4 ##################################')
snips = snips_4
plt_acc_loss(history_4_650, 'cnn_accuracy_Mod_4', 'cnn_loss_Mod_4_rgb', 'cnn_accuracy(100%).png', 'cnn_loss(1.9376e-06).png')

print("######################################################################")
# %%
print('######################################################################')
print('####################### models evaluation ############################')
print("######################################################################")
import numpy as np

print('######################### for Mod_1 ##################################')

# Mod 1 Vs ts1, tr1
def evaluate_model_1_ts1(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_ts1 = image_data_generator(test_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_ts1 = len(test_generator_1_ts1)
    predict_1_ts1 = model.evaluate_generator(test_generator_1_ts1, steps=nb_samples_ts1)

    return predict_1_ts1

def evaluate_model_1_tr1(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_tr1 = image_data_generator(train_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_tr1 = len(test_generator_1_tr1)
    predict_1_tr1 = model.evaluate_generator(test_generator_1_tr1, steps=nb_samples_tr1)

    return predict_1_tr1

# Mod 1 Vs ts3, tr3
def evaluate_model_1_ts3(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_ts3 = image_data_generator(test_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_ts3 = len(test_generator_1_ts3)
    predict_1_ts3 = model.evaluate_generator(test_generator_1_ts3, steps=nb_samples_ts3)

    return predict_1_ts3

def evaluate_model_1_tr3(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_tr3 = image_data_generator(train_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_tr3 = len(test_generator_1_tr3)
    predict_1_tr3 = model.evaluate_generator(test_generator_1_tr3, steps=nb_samples_tr3)

    return predict_1_tr3

# Mod 1 Vs ts4, tr4
def evaluate_model_1_ts4(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_ts4 = image_data_generator(test_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_ts4 = len(test_generator_1_ts4)
    predict_1_ts4 = model.evaluate_generator(test_generator_1_ts4, steps=nb_samples_ts4)

    return predict_1_ts4

def evaluate_model_1_tr4(model=None, filepath_1=None):
    if not model:
        model = models.load_model(filepath_1)
    test_generator_1_tr4 = image_data_generator(train_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_tr4 = len(test_generator_1_tr4)
    predict_1_tr4 = model.evaluate_generator(test_generator_1_tr4, steps=nb_samples_tr4)

    return predict_1_tr4

predict_1_tr1 = evaluate_model_1_tr1(model=model_1)
print('Mod_1_rgb_the accuracy of train data tr1 without data augment: (Known_Related)', predict_1_tr1[1])
predict_1_ts1 = evaluate_model_1_ts1(model=model_1)
print('Mod_1_rgb_the accuracy of test data ts1 without data augment: (Unknown_Related)', predict_1_ts1[1])

predict_1_tr3 = evaluate_model_1_tr3(model=model_1)
print('Mod_1_rgb_the accuracy of train data tr3 without data augment: (Known_UnRelated)', predict_1_tr3[1])
predict_1_ts3 = evaluate_model_1_ts3(model=model_1)
print('Mod_1_rgb_the accuracy of test data ts3 without data augment: (Unknown_UnRelated)', predict_1_ts3[1])

predict_1_tr4 = evaluate_model_1_tr4(model=model_1)
print('Mod_1_rgb_the accuracy of train data tr4 without data augment: (Known_UnRelated)', predict_1_tr4[1])
predict_1_ts4 = evaluate_model_1_ts4(model=model_1)
print('Mod_1_rgb_the accuracy of test data ts4 without data augment: (Unknown_UnRelated)', predict_1_ts4[1])


print('the accuracy of Mod_1_rgb_ Vs tr1 (  Known_  Related)', predict_1_tr1[1])
print('the accuracy of Mod_1_rgb_ Vs ts1 (Unknown_  Related)', predict_1_ts1[1])
print('the accuracy of Mod_1_rgb_ Vs tr3 (  Known_UnRelated)', predict_1_tr3[1])
print('the accuracy of Mod_1_rgb_ Vs ts3 (Unknown_UnRelated)', predict_1_ts3[1])
print('the accuracy of Mod_1_rgb_ Vs tr4 (  Known_UnRelated)', predict_1_tr4[1])
print('the accuracy of Mod_1_rgb_ Vs ts4 (Unknown_UnRelated)', predict_1_ts4[1])

print('the loss of Mod_1_rgb_ Vs tr1 (  Known_  Related)', predict_1_tr1[0])
print('the loss of Mod_1_rgb_ Vs ts1 (Unknown_  Related)', predict_1_ts1[0])
print('the loss of Mod_1_rgb_ Vs tr3 (  Known_UnRelated)', predict_1_tr3[0])
print('the loss of Mod_1_rgb_ Vs ts3 (Unknown_UnRelated)', predict_1_ts3[0])
print('the loss of Mod_1_rgb_ Vs tr4 (  Known_UnRelated)', predict_1_tr4[0])
print('the loss of Mod_1_rgb_ Vs ts4 (Unknown_UnRelated)', predict_1_ts4[0])

print('######################### for Mod_3 ##################################')

# Mod 3 Vs ts3, tr3
def evaluate_model_3_ts3(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_ts3 = image_data_generator(test_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_ts3 = len(test_generator_3_ts3)
    predict_3_ts3 = model.evaluate_generator(test_generator_3_ts3, steps=nb_samples_ts3)

    return predict_3_ts3

def evaluate_model_3_tr3(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_tr3 = image_data_generator(train_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_tr3 = len(test_generator_3_tr3)
    predict_3_tr3 = model.evaluate_generator(test_generator_3_tr3, steps=nb_samples_tr3)

    return predict_3_tr3

# Mod 3 Vs ts1, tr1
def evaluate_model_3_ts1(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_ts1 = image_data_generator(test_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_ts1 = len(test_generator_3_ts1)
    predict_3_ts1 = model.evaluate_generator(test_generator_3_ts1, steps=nb_samples_ts1)

    return predict_3_ts1

def evaluate_model_3_tr1(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_tr1 = image_data_generator(train_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_tr1 = len(test_generator_3_tr1)
    predict_3_tr1 = model.evaluate_generator(test_generator_3_tr1, steps=nb_samples_tr1)

    return predict_3_tr1

# Mod 3 Vs ts4, tr4
def evaluate_model_3_ts4(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_ts4 = image_data_generator(test_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_ts4 = len(test_generator_3_ts4)
    predict_3_ts4 = model.evaluate_generator(test_generator_3_ts4, steps=nb_samples_ts4)

    return predict_3_ts4

def evaluate_model_3_tr4(model=None, filepath_3=None):
    if not model:
        model = models.load_model(filepath_3)
    test_generator_3_tr4 = image_data_generator(train_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_tr4 = len(test_generator_3_tr4)
    predict_3_tr4 = model.evaluate_generator(test_generator_3_tr4, steps=nb_samples_tr4)

    return predict_3_tr4

predict_3_tr3 = evaluate_model_3_tr3(model=model_3)
print('Mod_3_rgb_the accuracy of train data tr3 without data augment: (Known_Related)', predict_3_tr3[1])
predict_3_ts3 = evaluate_model_3_ts3(model=model_3)
print('Mod_3_rgb_the accuracy of test data ts3 without data augment: (Unknown_Related)', predict_3_ts3[1])

predict_3_tr1 = evaluate_model_3_tr1(model=model_3)
print('Mod_3_rgb_the accuracy of train data tr1 without data augment: (Known_UnRelated)', predict_3_tr1[1])
predict_3_ts1 = evaluate_model_3_ts1(model=model_3)
print('Mod_3_rgb_the accuracy of test data ts1 without data augment: (Unknown_UnRelated)', predict_3_ts1[1])

predict_3_tr4 = evaluate_model_3_tr4(model=model_3)
print('Mod_3_rgb_the accuracy of train data tr4 without data augment: (Known_UnRelated)', predict_3_tr4[1])
predict_3_ts4 = evaluate_model_3_ts4(model=model_3)
print('Mod_3_rgb_the accuracy of test data ts4 without data augment: (Unknown_UnRelated)', predict_3_ts4[1])


print('the accuracy of Mod_3_rgb_ Vs tr3 (  Known_  Related)', predict_3_tr3[1])
print('the accuracy of Mod_3_rgb_ Vs ts3 (Unknown_  Related)', predict_3_ts3[1])
print('the accuracy of Mod_3_rgb_ Vs tr1 (  Known_UnRelated)', predict_3_tr1[1])
print('the accuracy of Mod_3_rgb_ Vs ts1 (Unknown_UnRelated)', predict_3_ts1[1])
print('the accuracy of Mod_3_rgb_ Vs tr4 (  Known_UnRelated)', predict_3_tr4[1])
print('the accuracy of Mod_3_rgb_ Vs ts4 (Unknown_UnRelated)', predict_3_ts4[1])

print('the loss of Mod_3_rgb_ Vs tr3 (  Known_  Related)', predict_3_tr3[0])
print('the loss of Mod_3_rgb_ Vs ts3 (Unknown_  Related)', predict_3_ts3[0])
print('the loss of Mod_3_rgb_ Vs tr1 (  Known_UnRelated)', predict_3_tr1[0])
print('the loss of Mod_3_rgb_ Vs ts1 (Unknown_UnRelated)', predict_3_ts1[0])
print('the loss of Mod_3_rgb_ Vs tr4 (  Known_UnRelated)', predict_3_tr4[0])
print('the loss of Mod_3_rgb_ Vs ts4 (Unknown_UnRelated)', predict_3_ts4[0])

print('######################### for Mod_4 ##################################')
# Mod 4 Vs ts4, tr4
def evaluate_model_4_ts4(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_ts4 = image_data_generator(test_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_ts4 = len(test_generator_4_ts4)
    predict_4_ts4 = model.evaluate_generator(test_generator_4_ts4, steps=nb_samples_ts4)

    return predict_4_ts4

def evaluate_model_4_tr4(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_tr4 = image_data_generator(train_dir_Mod4_200, batch_size=1, shuffle=False)

    nb_samples_tr4 = len(test_generator_4_tr4)
    predict_4_tr4 = model.evaluate_generator(test_generator_4_tr4, steps=nb_samples_tr4)

    return predict_4_tr4

# Mod 4 Vs ts3, tr3
def evaluate_model_4_ts3(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_ts3 = image_data_generator(test_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_ts3 = len(test_generator_4_ts3)
    predict_4_ts3 = model.evaluate_generator(test_generator_4_ts3, steps=nb_samples_ts3)

    return predict_4_ts3

def evaluate_model_4_tr3(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_tr3 = image_data_generator(train_dir_Mod3_200, batch_size=1, shuffle=False)

    nb_samples_tr3 = len(test_generator_4_tr3)
    predict_4_tr3 = model.evaluate_generator(test_generator_4_tr3, steps=nb_samples_tr3)

    return predict_4_tr3

# Mod 4 Vs ts1, tr1
def evaluate_model_4_ts1(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_ts1 = image_data_generator(test_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_ts1 = len(test_generator_4_ts1)
    predict_4_ts1 = model.evaluate_generator(test_generator_4_ts1, steps=nb_samples_ts1)

    return predict_4_ts1

def evaluate_model_4_tr1(model=None, filepath_4=None):
    if not model:
        model = models.load_model(filepath_4)
    test_generator_4_tr1 = image_data_generator(train_dir_Mod1_200, batch_size=1, shuffle=False)

    nb_samples_tr1 = len(test_generator_4_tr1)
    predict_4_tr1 = model.evaluate_generator(test_generator_4_tr1, steps=nb_samples_tr1)

    return predict_4_tr1

predict_4_tr4 = evaluate_model_4_tr4(model=model_4)
print('Mod_4_rgb_the accuracy of train data tr4 without data augment: (Known_Related)', predict_4_tr4[1])
predict_4_ts4 = evaluate_model_4_ts4(model=model_4)
print('Mod_4_rgb_the accuracy of test data ts4 without data augment: (Unknown_Related)', predict_4_ts4[1])

predict_4_tr1 = evaluate_model_4_tr1(model=model_4)
print('Mod_4_rgb_the accuracy of train data tr1 without data augment: (Known_UnRelated)', predict_4_tr1[1])
predict_4_ts1 = evaluate_model_4_ts1(model=model_4)
print('Mod_4_rgb_the accuracy of test data ts1 without data augment: (Unknown_UnRelated)', predict_4_ts1[1])

predict_4_tr3 = evaluate_model_4_tr3(model=model_4)
print('Mod_4_rgb_the accuracy of train data tr3 without data augment: (Known_UnRelated)', predict_4_tr3[1])
predict_4_ts3 = evaluate_model_4_ts3(model=model_4)
print('Mod_4_rgb_the accuracy of test data ts3 without data augment: (Unknown_UnRelated)', predict_4_ts3[1])


print('the accuracy of Mod_4_rgb_ Vs tr4 (  Known_  Related)', predict_4_tr4[1])
print('the accuracy of Mod_4_rgb_ Vs ts4 (Unknown_  Related)', predict_4_ts4[1])
print('the accuracy of Mod_4_rgb_ Vs tr1 (  Known_UnRelated)', predict_4_tr1[1])
print('the accuracy of Mod_4_rgb_ Vs ts1 (Unknown_UnRelated)', predict_4_ts1[1])
print('the accuracy of Mod_4_rgb_ Vs tr3 (  Known_UnRelated)', predict_4_tr3[1])
print('the accuracy of Mod_4_rgb_ Vs ts3 (Unknown_UnRelated)', predict_4_ts3[1])


print('the loss of Mod_4_rgb_ Vs tr4 (  Known_  Related)', predict_4_tr4[0])
print('the loss of Mod_4_rgb_ Vs ts4 (Unknown_  Related)', predict_4_ts4[0])
print('the loss of Mod_4_rgb_ Vs tr1 (  Known_UnRelated)', predict_4_tr1[0])
print('the loss of Mod_4_rgb_ Vs ts1 (Unknown_UnRelated)', predict_4_ts1[0])
print('the loss of Mod_4_rgb_ Vs tr3 (  Known_UnRelated)', predict_4_tr3[0])
print('the loss of Mod_4_rgb_ Vs ts3 (Unknown_UnRelated)', predict_4_ts3[0])

print('##################### All Evaluation Results #########################')
print('the accuracy of Mod_1_rgb_ Vs tr1 (  Known_  Related)=', predict_1_tr1[1])
print('the accuracy of Mod_1_rgb_ Vs ts1 (Unknown_  Related)=', predict_1_ts1[1])
print('the accuracy of Mod_1_rgb_ Vs tr3 (  Known_UnRelated)=', predict_1_tr3[1])
print('the accuracy of Mod_1_rgb_ Vs ts3 (Unknown_UnRelated)=', predict_1_ts3[1])
print('the accuracy of Mod_1_rgb_ Vs tr4 (  Known_UnRelated)=', predict_1_tr4[1])
print('the accuracy of Mod_1_rgb_ Vs ts4 (Unknown_UnRelated)=', predict_1_ts4[1])

print('the accuracy of Mod_3_rgb_ Vs tr3 (  Known_  Related)=', predict_3_tr3[1])
print('the accuracy of Mod_3_rgb_ Vs ts3 (Unknown_  Related)=', predict_3_ts3[1])
print('the accuracy of Mod_3_rgb_ Vs tr1 (  Known_UnRelated)=', predict_3_tr1[1])
print('the accuracy of Mod_3_rgb_ Vs ts1 (Unknown_UnRelated)=', predict_3_ts1[1])
print('the accuracy of Mod_3_rgb_ Vs tr4 (  Known_UnRelated)=', predict_3_tr4[1])
print('the accuracy of Mod_3_rgb_ Vs ts4 (Unknown_UnRelated)=', predict_3_ts4[1])

print('the accuracy of Mod_4_rgb_ Vs tr4 (  Known_  Related)=', predict_4_tr4[1])
print('the accuracy of Mod_4_rgb_ Vs ts4 (Unknown_  Related)=', predict_4_ts4[1])
print('the accuracy of Mod_4_rgb_ Vs tr1 (  Known_UnRelated)=', predict_4_tr1[1])
print('the accuracy of Mod_4_rgb_ Vs ts1 (Unknown_UnRelated)=', predict_4_ts1[1])
print('the accuracy of Mod_4_rgb_ Vs tr3 (  Known_UnRelated)=', predict_4_tr3[1])
print('the accuracy of Mod_4_rgb_ Vs ts3 (Unknown_UnRelated)=', predict_4_ts3[1])


print('the loss of Mod_1_rgb_ Vs tr1 (  Known_  Related)=', predict_1_tr1[0])
print('the loss of Mod_1_rgb_ Vs ts1 (Unknown_  Related)=', predict_1_ts1[0])
print('the loss of Mod_1_rgb_ Vs tr3 (  Known_UnRelated)=', predict_1_tr3[0])
print('the loss of Mod_1_rgb_ Vs ts3 (Unknown_UnRelated)=', predict_1_ts3[0])
print('the loss of Mod_1_rgb_ Vs tr4 (  Known_UnRelated)=', predict_1_tr4[0])
print('the loss of Mod_1_rgb_ Vs ts4 (Unknown_UnRelated)=', predict_1_ts4[0])

print('the loss of Mod_3_rgb_ Vs tr3 (  Known_  Related)=', predict_3_tr3[0])
print('the loss of Mod_3_rgb_ Vs ts3 (Unknown_  Related)=', predict_3_ts3[0])
print('the loss of Mod_3_rgb_ Vs tr1 (  Known_UnRelated)=', predict_3_tr1[0])
print('the loss of Mod_3_rgb_ Vs ts1 (Unknown_UnRelated)=', predict_3_ts1[0])
print('the loss of Mod_3_rgb_ Vs tr4 (  Known_UnRelated)=', predict_3_tr4[0])
print('the loss of Mod_3_rgb_ Vs ts4 (Unknown_UnRelated)=', predict_3_ts4[0])

print('the loss of Mod_4_rgb_ Vs tr4 (  Known_  Related)=', predict_4_tr4[0])
print('the loss of Mod_4_rgb_ Vs ts4 (Unknown_  Related)=', predict_4_ts4[0])
print('the loss of Mod_4_rgb_ Vs tr1 (  Known_UnRelated)=', predict_4_tr1[0])
print('the loss of Mod_4_rgb_ Vs ts1 (Unknown_UnRelated)=', predict_4_ts1[0])
print('the loss of Mod_4_rgb_ Vs tr3 (  Known_UnRelated)=', predict_4_tr3[0])
print('the loss of Mod_4_rgb_ Vs ts3 (Unknown_UnRelated)=', predict_4_ts3[0])
model_1.metrics_names
model_3.metrics_names
model_4.metrics_names
print("######################################################################")
