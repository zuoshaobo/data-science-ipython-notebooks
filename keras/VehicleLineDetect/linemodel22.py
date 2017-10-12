#!/usr/bin/env python
import time
import numpy as np
import tensorflow as tf
from  keras.models import load_model
from keras.callbacks import ModelCheckpoint
import random
import os
import cv2
import argparse
import json
import keras.callbacks as cbks
from keras import backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU,MaxPooling2D,Input,merge,UpSampling2D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import glob
from keras.layers.convolutional import Conv2D
batch_size=32
nb_epoch=150
smooth=1
def custom_loss(y_true, y_pred):
     #y_true = tf.Print(y_true, [y_true])
     #y_pred = tf.Print(y_pred, [tf.shape(y_pred)])
     loss1=mean_squared_error(y_true[:,0],y_pred[:,0])
     loss2=mean_squared_error(y_true[:,1],y_pred[:,1])
     return loss2+loss1
class CustomMetrics(cbks.Callback):
    def on_batch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('yss'):
                print logs
def iou_simple(actual, predicted):
    actual = K.flatten(actual)
    predicted = K.flatten(predicted)
    return K.sum(actual * predicted) / (1.0 + K.sum(actual) + K.sum(predicted))

def val_loss(actual, predicted):
    return -iou_simple(actual, predicted)
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
def test_model(model,trainx,trainy):

    def getpos(t):
        xfrom=float(t[0][0]*150)
        xfrom=int(xfrom)
        yfrom=150
        xto=int(xfrom-150.0/t[0][1])
        yto=0
        return xfrom,yfrom,xto,yto

    #'''
    video_path='/home/pub/Downloads/Advanced-Lane-Detection/project_video.mp4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset3/09110916_0062.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset3/09111926_0073.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset3/09120908_0087.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset3/09111929_0074.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset1/2017_0608_091436_292.MOV'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset1/2017_0608_085935_289.MOV'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset2/201706112044_000334AA.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset3/09111929_0074.MP4'
    video_path='/media/pub/b65b635d-0615-4f4a-922c-407fb777a2f3/datasets/video_datasets/dataset2/201706120819_000361AA.MP4'
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
	frame=cv2.resize(frame,(300,300),interpolation=cv2.INTER_CUBIC)
        crop_img = frame[150:300, 0:300] 
        crop_img=crop_img/255.0
        t2=time.time()
        t1=model.predict(np.expand_dims(crop_img, axis=0))
        print time.time()-t2
        img2=np.zeros_like(t1[0])
        img2[t1[0]>0.5]=1.0
        segmented = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        segmented[:,:,0:1] = 0
        crop_img=np.array(crop_img,dtype=np.float32)
        image = cv2.addWeighted(segmented, 0.5, crop_img, 0.5, 0.0)
        cv2.imshow('2',image)
        #cv2.imshow('3',t1[0])
        cv2.waitKey(1)
    #'''



    '''
    for img in trainx:
        t=model.predict(np.expand_dims(img, axis=0))
        img2=np.zeros_like(t[0])
        img2[t[0]>0.1]=1.0
        cv2.imshow('1',img)
        cv2.imshow('2',img2)
        segmented = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        segmented[:,:,0:1] = 0
        image = cv2.addWeighted(segmented, 0.5, img, 0.5, 0.0)
        cv2.imshow('3',image)
        cv2.waitKey(0)
    '''
    

    exit(0)
def get_model():

    row=150
    col=300
    ch=3
    inputs = Input((row, col,3))
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(3, 3))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)


    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    #------------------------------------------------------
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv4)

    #'''
    up7 = merge([UpSampling2D(size=(3, 3))(conv4), conv2], mode='concat', concat_axis=3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv1], mode='concat', concat_axis=3)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(up8)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = Conv2D(1, 1, 1, activation='sigmoid')(conv7)
    #'''

    model = Model(input=inputs, output=conv8)
    
    adam=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy')
    return model

model=get_model()
model.summary()
#exit(0)
'''
X_train=[]
Y_train=[]
txtfiles=glob.glob('datasets_store/*_g.jpg')
for t in txtfiles:
    imagename1=t
    imagename2=t.split('_g')[0]
    img1=cv2.imread(imagename1,0)
    img11=np.zeros_like(img1)
    img11[img1>100]=255
    X_train.append(cv2.imread(imagename2))
    Y_train.append(img11)
    #cv2.imshow('',cv2.imread(imagename2))
    #cv2.imshow('1',img11)
    cv2.waitKey(0)
txtfiles=glob.glob('other/*_g.jpg')
for t in txtfiles:
    imagename1=t
    imagename2=t.split('_g')[0]
    img1=cv2.imread(imagename1,0)
    img11=np.zeros_like(img1)
    img11[img1>100]=255
    X_train.append(cv2.imread(imagename2))
    Y_train.append(img11)
    cv2.imshow('',cv2.imread(imagename2))
    cv2.imshow('1',img11)
    cv2.waitKey(0)



X_train=np.array(X_train,dtype=np.float32)
Y_train=np.array(Y_train,dtype=np.float32)
Y_train=np.expand_dims(Y_train,axis=3)
print X_train.shape
print Y_train.shape
X_train=X_train/255.0
Y_train=Y_train/255.0

'''
#print Y_train
import keras
#keras.losses.custom_loss =val_loss
model=load_model('line3.best.hdf5')
test_model(model,None,None)

def generate_arrays_from_file2(path):
    index=0
    while 1:
            X_train=np.zeros((64,300,600,3))
            Y_train=np.zeros((64,300,600,1))
            print index
            index+=1
            yield (X_train, Y_train)


def generate_arrays_from_file(path):
    txtfiles=glob.glob('datasets_store_chaoge/*_g.jpg')
    txtfiles3=glob.glob('datasets_store_tangsong/*_g.jpg')
    txtfiles2=glob.glob('other/*_g.jpg')
    txtfiles.extend(txtfiles2)
    txtfiles.extend(txtfiles3)
    while 1:
        index=0
        random.shuffle(txtfiles)
        print 'train sampels is :',len(txtfiles)
        while (index+batch_size)<len(txtfiles):
            #print 'here',index
            X_train=[]
            Y_train=[]
            for t in txtfiles[index:index+batch_size]:
                imagename1=t
                imagename2=t.split('_g')[0]
                #print imagename1
                #print imagename2
                img1=cv2.imread(imagename1,0)
                img1=cv2.resize(img1,(300,150))
                img11=np.zeros_like(img1)
                img11[img1>100]=255

                X_train.append(cv2.resize(cv2.imread(imagename2),(300,150)))
                Y_train.append(img11)

            X_train=np.array(X_train,dtype=np.float32)
            Y_train=np.array(Y_train,dtype=np.float32)
            Y_train=np.expand_dims(Y_train,axis=3)
            X_train=X_train/255.0
            Y_train=Y_train/255.0
            #print X_train.shape
            #print Y_train.shape
            #X_train=np.zeros((64,300,600,3))
            #Y_train=np.zeros((64,300,600,1))
            #exit(0)
            index+=batch_size
            yield (X_train, Y_train)
'''
model.fit(X_train, Y_train,
        batch_size=batch_size, nb_epoch=nb_epoch,
        verbose=1)
'''
'''
p=generate_arrays_from_file('')
while 1:
    x,y=p.next()
    print x.shape
    print y.shape
'''
filepath="line3.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit_generator(generate_arrays_from_file(''),
                steps_per_epoch=579,epochs=nb_epoch,verbose=1,callbacks=callbacks_list)
model.save('line3.h5')
t=model.predict(np.expand_dims(X_train[0], axis=0))
