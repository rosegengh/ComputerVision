import cv2
import os
import exifread
from imutils import paths
import argparse
import math
from PIL import Image, ImageStat
from exif import Image
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil




#im = cv2.imread('Test/im1.jpg')
#print(im.shape)
imagegen = ImageDataGenerator()

train = imagegen.flow_from_directory("Image_train_apeature/", class_mode="categorical", shuffle=False, batch_size=100, target_size=(64, 64))
val = imagegen.flow_from_directory("Image_val_apeature/", class_mode="categorical", shuffle=False, batch_size=100, target_size=(64, 64))



train_iso = imagegen.flow_from_directory("Image_train_iso/", class_mode="categorical", shuffle=False, batch_size=100, target_size=(64, 64))
val_iso = imagegen.flow_from_directory("Image_val_iso/", class_mode="categorical", shuffle=False, batch_size=100, target_size=(64, 64))

a = iter(train).next()[0]
print(a.shape)



def classify_image():
    path_exif = 'exif'
    path_image = 'Test'
    apeature = []
    shutter_speed = []
    iso = []
    count   = 0 
    parent_dir_apeature = 'Image_train_iso'
    parent_dir_iso = 'Image_train_iso'
    for filename in os.listdir(path_image):
        temp= filename[2:]
        exif_name = temp[:-4] +'.txt'
        
        exif_txt = os.path.join(path_exif,exif_name)
        if os.path.isfile(exif_txt):
            f = open(exif_txt,'rb')
            lines = f.readlines()
            if(len(lines) > 0):
                for line in lines:
                    
                    # if ('-Aperture' in str(line)):
                    #     index = lines.index(line)
                    #     ApertureTemp = lines[index + 1][:-2]
                    #     Afolder = str(ApertureTemp).replace("/","")
                    #     src = path_image +'/' + filename

                    #     dst = parent_dir_apeature + '/'  + Afolder


                    #     shutil.copy(src, dst)
                    #     apeature.append(lines[index + 1][:-2])
                    #     continue
                
                    if('-ISO Speed' in str(line)):
                        count = count + 1
                        index = lines.index(line)
                        isoTemp = lines[index+1][:-2]
                        if('May be' in str(isoTemp)):
                            isoTemp = isoTemp[:-53]
                        if(',' in str(isoTemp)):
                            temp = isoTemp.split()
                            isoTemp = temp[0][:-1]
                        if('Error' in str(isoTemp)):
                            isoTemp = '100'

                        Ifolder = str(isoTemp)
                        src = path_image +'/' + filename
                        dst = parent_dir_apeature + '/'  + Ifolder
                        shutil.copy(src, dst)
                        iso.append(isoTemp)


                        continue
                        #print(isoTemp)

                    # if('-Exposure' in str(line)):
                    #     index = lines.index(line)
                    #     shutterTemp = lines[index+1][:-2]
                    #     continue
                        #print(shutterTemp)
            f.close()
    #apeature = list(dict.fromkeys(apeature))

    # for index in apeature:
    #     folder = str(index).replace("/","")
    #     path = parent_dir_apeature + '/' + folder
        
    #     os.mkdir(path)
    # iso = list(dict.fromkeys(iso))

    # for index in iso:
    #     #folder = str(index).replace("/","")
    #     path = parent_dir_iso + '/' + str(index)
        
    #     os.mkdir(path)
    
    


#classify_image()
  

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])      

def build_cnn():
    model = Sequential()
    model.add(InputLayer(input_shape=(64, 64, 3)))

    # 1st conv block
    model.add(Conv2D(25, (5, 5), activation='relu', strides=(1, 1), padding='same', input_shape = (64,64,3)))
    
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    # 2nd conv block
    model.add(Conv2D(50, (5, 5), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())
    # 3rd conv block
    model.add(Conv2D(70, (3, 3), activation='relu', strides=(2, 2), padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    # ANN block
    model.add(Flatten())
    model.add(Dense(units=21, activation='relu'))
    model.add(Dense(units=21, activation='softmax'))
    model.add(Dropout(0.25))
    # output layer
    # model.add(Flatten())
    # model.add(Dense(units=236, activation='softmax'))
    print("finish generate the CNN model")

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # fit on data for 30 epochs
    print(model.summary())
    history = model.fit_generator(train_iso, epochs=10,validation_data=val_iso)

    acc =history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,10.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return model


def Imagenet():
    IMG_SIZE = (64, 64)
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

    image_batch, label_batch = next(iter(train))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(64, 64, 3))
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()
    history = model.fit_generator(train_iso, epochs=10,validation_data=val_iso)

    acc =history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()




#build_cnn()
Imagenet()

