#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 22:36:03 2018

@author: sidnpoo
"""
import os
from tqdm import tqdm
import glob
import cv2
import imutils
import numpy as np

def create_train_val_test_dirs(base_dir, class_name_list):   

    # Directories for our training,
    # validation and test splits
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    class_folder_list={}
    for item in class_name_list:

        train_class_dir = os.path.join(train_dir, str(item))
        print(train_class_dir)
        if not os.path.exists(train_class_dir):
            os.mkdir(train_class_dir)

        val_class_dir = os.path.join(validation_dir,str(item))
        print(val_class_dir)
        if not os.path.exists(val_class_dir):
            os.mkdir(val_class_dir)

        test_class_dir = os.path.join(test_dir, str(item))
        print(test_class_dir)
        if not os.path.exists(test_class_dir):
            os.mkdir(test_class_dir)
        
        class_folder_list[item]= [{"train":train_class_dir}, {"validation": val_class_dir}, {"test":test_class_dir}]
    return class_folder_list


def split_dataset(split_ratios, file_list):
 
    train=split_ratios[0]*len(file_list)
    validation=split_ratios[2]*len(file_list)     
    test=split_ratios[1]*len(file_list)
    print(len(file_list))
    assert(len(file_list) == train+test+validation)
    train_list = file_list[:int(train)]
    print(len(train_list))
    validation_list = file_list[int(train):int(train+validation)]
    print(len(validation_list))
    test_list = file_list[int(train+validation):]
    print(len(test_list))
    print("--")
    return train_list, validation_list,test_list

import shutil


def copy_files(fromdir, todir, file_list):
    for file in tqdm(file_list):
        srcfile=os.path.join(fromdir, file)
        destfile=os.path.join(todir, file)
        #print("Copying",srcfile,destfile)
        shutil.copy(srcfile,destfile)



def prepare_dataset_as_nparray(class_path, file_pattern,  resize_width, resize_height,label_value):
    X = []
    y = []
    class_list = glob.glob((class_path+str(os.sep)+file_pattern))
 
    
    data = [(path, label_value) for path in class_list]
    for path, label in data:
        print("Processing ..",path)
        #img = cv2.imread(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = imutils.resize(img, width=resize_width, height=resize_height)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)




def normalize_dataset(X, y):
    X = X.astype(np.float32) / 255.
    y = y.astype(np.int32)
    #print X.dtype, X.min(), X.max(), X.shape
    #print y.dtype, y.min(), y.max(), y.shape
    return X,y
    

import matplotlib.pyplot as plt

def plot_loss_and_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
    
