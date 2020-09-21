#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:00:31 2020

@author: jesse
"""

#Importing necessary tools
import tensorflow.compat.v1 as tf_compat
import tensorflow as tf
import tensorflow.keras.backend as K
import math
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#import sklearn.metrics
import matplotlib.pyplot as plt
import segmentation_models as sm
sm.set_framework('tf.keras')
import glob
import numpy as np
import os
import h5py
import re
tf.enable_eager_execution()

#Defining generator that yields paths for random images + label data
def gen(datasettype):
   #takes the paths of the images and label images, encodes it and zip the appropriate files together 
   dataset_used = 'PASCAL' #PASCAL or LIP
   if dataset_used == 'PASCAL':
       if datasettype == 'train':
           path = '/home/jesse/HumanSegmentation/pascal_person_part/train_part_JPEGs/*'
           label = '/home/jesse/HumanSegmentation/pascal_person_part/train_part_segments_gray/*'
       elif datasettype == 'validation':
           path = '/home/jesse/HumanSegmentation/pascal_person_part/val_part_JPEGs/*'
           label = '/home/jesse/HumanSegmentation/pascal_person_part/val_part_segments_gray/*'
      #elif datasettype == 'test':
          #path = 
          #path2 =  
       #images_path=glob.glob(path)
       #images_path = sorted(images_path, key=lambda x:float(re.findall("(\d+)",x)[0]))
       images_path = sorted(glob.glob(path))
       #base = os.path.basename(images_path) 
       #print(base)
       #one, two = os.path.splitext(base)
       images_path=[x.encode('utf-8') for x in images_path] 
       #print(one)
       #labels_path=glob.glob(label)
       labels_path = sorted(glob.glob(label))
       #labels_path = sorted(labels_path, key=lambda x:float(re.findall("(\d+)",x)[0]))
       labels_path=[x.encode('utf-8') for x in labels_path]
       images_and_labels = zip(images_path,labels_path)
       for paths in images_and_labels:
           yield paths

   elif dataset_used == 'LIP':
       if datasettype == 'train':
           path = '/home/jesse/HumanSegmentation/LIP/TrainVal_images/TrainVal_images/train_images/*'
       elif datasettype == 'validation':
           path = '/home/jesse/HumanSegmentation/LIP/TrainVal_images/TrainVal_images/val_images/*'
       #elif datasettype == 'test':
           #path = 
           #path2 =
       images_path=glob.glob(path)  
       for path in images_path:
           base = os.path.basename(path) 
           one, two = os.path.splitext(base)
           if datasettype == 'train':
               label = '/home/jesse/HumanSegmentation/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations_gray/'+one+'.png'
           elif datasettype == 'validation':
               label = '/home/jesse/HumanSegmentation/LIP/TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations_gray/'+one+'.png'
       images_path=[x.encode('utf-8') for x in images_path]
       labels_path=glob.glob(label)
       labels_path=[x.encode('utf-8') for x in labels_path]
       images_and_labels = zip(images_path,labels_path)
       for paths in images_and_labels:
           yield paths 
  #labels_path=glob.glob(path2)
  #labels_path=[x.encode('utf-8') for x in labels_path]
  #couples=zip(images_path,labels_path)
  #for paths in couples:
  #    print(paths)
  #    yield paths   

def read_image_and_label(image_path,label_path):
  #the mapping function, first the paths will be read and the images saved as tensors. Then the tensors will be normalized in [0,1]
  image_tensor=tf.image.decode_image(tf.io.read_file(image_path))
  image_tensor=tf.math.divide(image_tensor,255) #normalize
  image_tensor = tf.cast(image_tensor, tf.float32)
  #Resizing images for faster training
  #image_tensor=tf.image.resize_image_with_crop_or_pad(image_tensor,400,500)
  image_tensor=tf.expand_dims(image_tensor,axis=0)
  image_tensor=tf.image.resize_bilinear(image_tensor,(250,250))
  image_tensor=tf.squeeze(image_tensor,axis=0)
  label_tensor=tf.image.decode_image(tf.io.read_file(label_path))
  #label_tensor=tf.math.divide(image_tensor,255) #normalize
  #label_tensor=tf.image.resize_image_with_crop_or_pad(label_tensor,400,500)
  label_tensor=tf.expand_dims(label_tensor,axis=0)
  label_tensor=tf.image.resize_bilinear(label_tensor,(250,250))
  label_tensor=tf.squeeze(label_tensor,axis=0)
  label_tensor = tf.cast(label_tensor, tf.int32)
  #Setting unnecessary classes from cityscapes to 0, to reduce total classes to 20
  #label_tensor =tf.gather(new_cids, label_tensor)
  return image_tensor,label_tensor

def input_fn(batch_size_,datasettype):
  #make the dataset usable
  buffer_size_ = 100
  dataset = tf.data.Dataset.from_generator(lambda: gen(datasettype),(tf.string,tf.string))
  dataset = dataset.shuffle(buffer_size=buffer_size_,reshuffle_each_iteration=True).repeat(count=None)  #Shuffle to mix up the order of data training per epoch
  dataset = dataset.map(read_image_and_label)   #Mapping the dataset to a tensor
  dataset = dataset.batch(batch_size_)  #Batching the dataset for training
  dataset = dataset.prefetch(None)
  return dataset

def test_data(batch_size_,datasettype):
  #test if the data is what is expected
  dataset=input_fn(batch_size_,'train')
  it= dataset.make_one_shot_iterator()
  next_element=it.get_next()
  for i in range(batch_size_):
    image,label= next_element
    image=image[i,:,:,:]
    image=tf.cast(image, tf.float32)
    label=label[i,:,:,:]
    label=tf.cast(label, tf.float32)
    #Changing color values of the classes so they allign with the cityscape labe palette
    #palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[50,20,30]]
    #label=tf.gather(palette, label)
    label = tf.squeeze(label)
    plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.subplot(122)
    plt.imshow(label)
    plt.show()

#[void,road,sidewalk,building,wall,fence,pole,traffic light,traffic sign,vegetation,terrain,sky,person,rider,car,truck,bus,train,motorcycle,bicycle]
  #Defining the metric Intersection over Union
def sparse_Mean_IOU(y_true, y_pred, class_id):
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    i=class_id
    true_labels = K.equal(y_true[:,:,0], i)
    pred_labels = K.equal(pred_pixels, i)
    inter = tf.cast(true_labels & pred_labels, tf.float32)
    union = tf.cast(true_labels | pred_labels, tf.float32)
    sumlab = tf.cast(true_labels, tf.float32)
    
    legal_batches = K.sum(sumlab, axis=1)>0
    if (K.sum(union, axis=1)) == 0:
        ious = 0
    else:
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
    iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return iou

def sparse_Mean_IOU_formean(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, tf.float32)
        union = tf.cast(true_labels | pred_labels, tf.float32)
        sumlab = tf.cast(true_labels, tf.float32)
        legal_batches = K.sum(sumlab, axis=1)>0
        if K.sum(union, axis=1) == 0:
            ious = 0
        else:
            ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

  #Calculating IOUs for every class	
def fn1(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,0)	
def fn2(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,1)	
def fn3(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,2)	
def fn4(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,3)	
def fn5(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,4)	
def fn6(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,5)	
def fn7(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,6)	
def fn8(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,8)	
def fn9(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,9)	
def fn10(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,10)	
def fn11(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,11)
def fn12(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,12)	
def fn13(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,13)	
def fn14(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,14)	
def fn15(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,15)	
def fn16(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,16)	
def fn17(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,17)	
def fn18(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,18)	
def fn19(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,19)	
	
def val_fn1(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,0)	
def val_fn2(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,1)	
def val_fn3(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,2)	
def val_fn4(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,3)	
def val_fn5(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,4)	
def val_fn6(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,5)	
def val_fn7(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,6)	
def val_fn8(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,8)	
def val_fn9(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,9)	
def val_fn10(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,10)	
def val_fn11(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,11)
def val_fn12(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,12)	
def val_fn13(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,13)	
def val_fn14(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,14)	
def val_fn15(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,15)	
def val_fn16(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,16)	
def val_fn17(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,17)	
def val_fn18(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,18)	
def val_fn19(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,19)

def TheModel(batch_size_,restore):
  
  dataset=input_fn(batch_size_,datasettype = 'train')  
  datasetval = input_fn(batch_size_,datasettype = 'validation') 
  
  train_set = 'PASCAL'
  set_partition = 1
  if train_set == 'LIP':
      t_steps = int(set_partition*(30462/batch_size_))
      v_steps = int(set_partition*(10000/batch_size_))
  elif train_set == 'PASCAL':
      t_steps = int(set_partition*(1716/batch_size_))
      v_steps = int(set_partition*(1817/batch_size_))
  
  BACKBONE = 'efficientnetb3'
  CLASSES = ['background', 'torso','head','arms','hands','legs','feet']
  LR = 0.0001
  EPOCHS = 10

  # define network parameters
  n_classes = (len(CLASSES) + 1)  # case for binary and multiclass segmentation
  activation = 'softmax'
  
  optim = k.optimizers.Adam(LR)
  #create model
  model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

  # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
  # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
  dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.3, 1, 1, 1, 1, 1, 1])) 
  focal_loss = sm.losses.CategoricalFocalLoss()
  total_loss = dice_loss + (1 * focal_loss)

  # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
  # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

  metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

  # compile keras model with defined optimozer, loss and metrics
  model.compile(optim, total_loss, metrics)
  customcallbacks = [
    k.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    k.callbacks.ReduceLROnPlateau(),
  ]


  # train model
  history = model.fit_generator(
    dataset, 
    steps_per_epoch=t_steps, 
    epochs=EPOCHS, 
    callbacks=customcallbacks, 
    validation_data=datasetval, 
    validation_steps=v_steps,
  )
  
  
  # Plot training & validation iou_score values
  plt.figure(figsize=(30, 5))
  plt.subplot(121)
  plt.plot(history.history['iou_score'])
  plt.plot(history.history['val_iou_score'])
  plt.title('Model iou_score')
  plt.ylabel('iou_score')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')

  # Plot training & validation loss values
  plt.subplot(122)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()
#  #create model
#  model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
#  #Get training dataset
#  dataset=input_fn(batch_size_,datasettype = 'train')
#  #Get Validation dataset
#  datasetval = input_fn(batch_size_,datasettype = 'validation')
#  #Defining the resulting metrics
#  metrics =[fn1,fn2,fn3,fn4,fn5,fn6,fn7]
#  train_set = 'PASCAL' # LIP or PASCAL
#  set_partition = 1
#  if train_set == 'LIP':
#      t_steps = int(set_partition*(30462/batch_size_))
#      v_steps = int(set_partition*(10000/batch_size_))
#  elif train_set == 'PASCAL':
#      t_steps = int(set_partition*(1716/batch_size_))
#      v_steps = int(set_partition*(1817/batch_size_))
#  #Restoring model possibility
#  if restore == True:
#      model = tf.contrib.saved_model.load_keras_model('./saved_models/1561026721')
#  else:
#      #The Model with Resnet50, BatchNorm, SGD optimizer and Categorical Cross-entropy Loss
#      model=tf.keras.models.Sequential()
#      inputs=tf.keras.layers.Input(shape=(250,250,3))
#      model.add(ResNet50(include_top=False, weights='imagenet',input_tensor=inputs, pooling=None, classes = 7))
#      model.add(tf.keras.layers.Conv2D(7,(1,1)))
#      model.add(layers.BatchNormalization())   
#      model.add(tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x,(250,250))))
#      model.summary()
#  learning_rate = 0.002
#  sgd = tf.keras.optimizers.SGD(learning_rate, momentum=0.8, nesterov=True)
#  model.compile(optimizer=sgd,   #SGD with momemtum usually gives good enough results without too many parameters as is the case for ADAMoptimizer
#                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),  
#                metrics=metrics)  
#  history=model.fit(dataset,validation_data = datasetval, epochs=10, steps_per_epoch=t_steps,validation_steps=v_steps) # 2975 Training Images - 1525 Test Images - 500 validation Images
#  #Saving the model after training
#  saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models/saved_model_test7bigsparse")
  
  return model

# Testing the model
def test_model(batch_size_,restore):
  #test if the output is what is expected; prints model generated images of the last batch
  dataset=input_fn(batch_size_,datasettype = 'train')
  #datasetval = input_fn(batch_size_,datasettype = 'validation')
  #model=TheModel(batch_size_,restore)
  model = tf.contrib.saved_model.load_keras_model('./saved_models/saved_model_test7bigsparse')
  it=dataset.make_one_shot_iterator()
  mean_iou_cat=0
  iou_cat = 0
  mean_iou_c = [0,0,0,0,0,0,0]
  for i in range(batch_size_):
    next_element=it.get_next()
    image_original,label_original=next_element
    output = model.predict_on_batch(image_original)
    pred = output[i,:,:,:]
    #Argmax to find the class with highest probability
    pred=tf.keras.backend.argmax(pred,axis=-1)
    #palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]]
    #palette = [[0,0,0],[20,20,20],[40,40,40],[60,60,60],[80,80,80],[100,100,100],[120,120,120]]
    #pred=tf.gather(palette, pred)   
    #image_original = image_original[i,:,:,:]*255
    #image_original=tf.cast(image_original, tf.int32)
    label=label_original[i,:,:,:]
    #label=tf.gather(palette, label)
    label=tf.squeeze(label)
    iou=np.empty(0)
    for j in range(7):
      iou_cat=sparse_Mean_IOU(label_original,output,j)
      
      if iou_cat.shape == (0,1):
          iou_cat = 0
      iou=np.append(iou,iou_cat) 
      iou=iou.flatten()
    mean_iou_c= mean_iou_c+ iou
    print(mean_iou_c)
    mean_iou=np.mean(iou)
    mean_iou_cat=mean_iou_cat+mean_iou
    #print(mean_iou_cat)
    #print('iou, mean iou')
    #print(iou, mean_iou)
    #Showing prediction labels
  #mean_iou_batch1=mean_iou_cat/batch_size_
  #print('mean iou batch1')
  #print(mean_iou_batch1)
  mean_iou_check = sparse_Mean_IOU_formean(label_original,pred)
  print('mean_iou',mean_iou_check)
  mean_iou_per_class = np.divide(mean_iou_c,batch_size_)
  print('mean iou per class',mean_iou_per_class)
  #b = np.count_nonzero(mean_iou_per_class)
  #mean_iou_batch = np.sum(mean_iou_per_class)/b
  #print(mean_iou_batch)
  image_original = tf.squeeze(image_original)
  pred = tf.squeeze(pred)
  label = tf.squeeze(label)
  #im1 = plt.imshow(image_original)
  #plt.show()
  
  f, (ax1, ax2, ax3) = plt.subplots(3, sharex=False, sharey=False)
  ax1.imshow(image_original)
  ax2.imshow(label)
  ax3.imshow(pred)
  

#Main Function
if __name__=='__main__':
  #Choosing between loading or saving the model
  restore = False 
  batch_size_ = 1
  #test_data(batch_size_,datasettype = 'train')
  #TheModel(batch_size_,restore)
  test_model(batch_size_,restore)
