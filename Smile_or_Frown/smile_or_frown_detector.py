#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 00:10:33 2018

@author: skhimsar
"""

import os
from myutils import *


# The path to the directory where the original
# dataset was uncompressed
original_smiles_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/positives/positives7'
original_frowns_dataset_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/SMILEs/negatives/negatives7'

# The directory where we will
# store our smaller dataset
base_dir = '/Users/sidnpoo/Downloads/MLStuff_DoNotDelete/_DATASETS_/SmileFrownData/dataset'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    
    
folder_list = create_train_val_test_dirs(base_dir, ["smiles","frowns"])

import random

random.seed(10)

frown_files = [ f for f in os.listdir(original_frowns_dataset_dir)]
random.shuffle(frown_files)

smile_files = [ f for f in os.listdir(original_smiles_dataset_dir)]
random.shuffle(smile_files)

print(len(smile_files))
print(len(frown_files))

TRAIN=0.8
VALIDATION=0.1
TEST=0.1

#split the total dataset into 3 parts for each class.
frown_train,frown_val,frown_test = split_dataset((TRAIN,VALIDATION,TEST), frown_files)
smile_train,smile_val,smile_test = split_dataset((TRAIN,VALIDATION,TEST), smile_files)

#copy them into respective train val and test folders per class
copy_files(original_frowns_dataset_dir, folder_list["frowns"][0]["train"], frown_train)
copy_files(original_frowns_dataset_dir, folder_list["frowns"][1]["validation"], frown_val)
copy_files(original_frowns_dataset_dir, folder_list["frowns"][2]["test"], frown_test)

copy_files(original_smiles_dataset_dir, folder_list["smiles"][0]["train"], smile_train)
copy_files(original_smiles_dataset_dir, folder_list["smiles"][1]["validation"], smile_val)
copy_files(original_smiles_dataset_dir, folder_list["smiles"][2]["test"], smile_test)


print('total training smile images:', len(os.listdir(folder_list["smiles"][0]["train"])))
print('total training smile images:', len(os.listdir(folder_list["smiles"][1]["validation"])))
print('total training smile images:', len(os.listdir(folder_list["smiles"][2]["test"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][0]["train"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][1]["validation"])))
print('total training frown images:', len(os.listdir(folder_list["frowns"][2]["test"])))


train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,  BatchNormalization,  Activation
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

INPUT_IMAGE_SIZE=64
COLOR_DEPTH_DIM=1

# model 1 --------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=32,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=32,
        color_mode='grayscale',
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break




model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
                        input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_DEPTH_DIM)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=10,
      validation_data=validation_generator,
      validation_steps=50)

plot_loss_and_accuracy(history)

model.save("smile_or_frown_model1.h5")

# model 2 --------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)


# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
        batch_size=20,
        color_mode='grayscale',
        class_mode='categorical')



from keras.optimizers import SGD

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3),  padding='same',activation='relu', 
                 input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, COLOR_DEPTH_DIM)))
model.add(Conv2D(32, (3, 3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
sgd = SGD(lr=0.0004, decay=1e-6, momentum=0.9, nesterov=True)
#adam=keras.optimizers.Adam(lr=0.0000005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

    # Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(train_generator,
                      validation_data=validation_generator,
                        epochs=20, 
                        verbose=1,
                        steps_per_epoch=50)


plot_loss_and_accuracy(history)

model.save("smile_or_frown_model2.h5")

