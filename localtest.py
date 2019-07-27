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
import sklearn.metrics
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
tf.enable_eager_execution()

#Defining generator that yields paths for random images + label data
def gen(datasettype):
   #takes the paths of the images and label images, encodes it and zip the appropriate files together  
  if datasettype == 'train':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*'
      path2 = "D:/s141533/NeuralNetworks/TrainingSetCityScapes/gtFine/train/*/*labelIds*"  
  elif datasettype == 'validation':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/leftImg8bit/val/*/*'
      path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/gtFine/val/*/*labelIds*'
  elif datasettype == 'test':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/leftImg8bit/train/bremen/*'
      path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/gtFine/train/bremen/*labelIds*' 
  elif datasettype == 'augment':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/NewestAugmentedImages/augmentation/*/image*'
      path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/NewestAugmentedImages/augmentation/*/label*' 
  elif datasettype == 'testwild':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/WilddashTesting2/wd_val_01/*0.png'
      path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/WilddashTesting2/wd_val_01/*labelIds*' 
  elif datasettype == 'testwildweather':
      path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/WilddashTestingWeather/*0.png'
      path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/WilddashTestingWeather/*labelIds*'
  #path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit*'
  #path2 = "D:/s141533/NeuralNetworks/TrainingSetCityScapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds*"
  
  #path = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/Testimg/*'
  #path2 = 'D:/s141533/NeuralNetworks/TrainingSetCityScapes/Testlabels/*'
  #path='kaasbomber/deep-learning/TrainingSetCityScapes/leftImg8bit_trainvaltest/leftImg8bit/train/b*/*' 
  #path2='kaasbomber/deep-learning/TrainingSetCityScapes/gtFine/train/b*/*labelIds*' 
  
  #path='/home/kaasbomber/deep-learning/TrainingSetCityScapes/leftImg8bit/train/b*/*' 
  #path2='/home/kaasbomber/deep-learning/TrainingSetCityScapes/gtFine/train/b*/*labelIds*'
# =============================================================================
#   if datasettype == 'train':
#       path='C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test/*'
#       path2='C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test_label/*'
#   elif datasettype == 'validation':
#       path='C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test/*'
#       path2='C:/Users/s141010/Documents/school/master/CNN/leftImg8bit/test/test_label/*'
# =============================================================================
      
  images_path=glob.glob(path)#glob.glob(path+"/leftImg8bit/train/aachen/*")
  images_path=[x.encode('utf-8') for x in images_path]  
  labels_path=glob.glob(path2)#glob.glob(path+"/gtFine_trainvaltest/gtFine/train/aachen/aachen_[]_gtFine_labellds.png")
  labels_path=[x.encode('utf-8') for x in labels_path]
  couples=zip(images_path,labels_path)
  for paths in couples:
      yield paths   

def read_image_and_label(image_path,label_path):
  #the mapping function, first the paths will be read and the images saved as tensors. Then the tensors will be normalized in [0,1]
  image_tensor=tf.image.decode_image(tf.io.read_file(image_path))
  image_tensor=tf.math.divide(image_tensor,255) #normalize
  image_tensor = tf.cast(image_tensor, tf.float32)
  #Resizing images for faster training
  image_tensor=tf.image.resize_image_with_crop_or_pad(image_tensor,256,512)
  label_tensor=tf.image.decode_image(tf.io.read_file(label_path))
  #label_tensor=tf.math.divide(image_tensor,255) #normalize
  label_tensor=tf.image.resize_image_with_crop_or_pad(label_tensor,256,512)
  label_tensor = tf.cast(label_tensor, tf.int32)
  #Setting unnecessary classes from cityscapes to 0, to reduce total classes to 20
  new_cids =[0,0,0,0,0,0,0,1,2,0,0,3,4,5,0,0,0,6,0,7,8,9,10,11,12,13,14,15,16,0,0,17,18,19,0]
  label_new =tf.gather(new_cids, label_tensor)
  return image_tensor,label_new

def input_fn(batch_size_,datasettype):
  #make the dataset usable
  buffer_size_ = 10
  dataset = tf.data.Dataset.from_generator(lambda: gen(datasettype),(tf.string,tf.string))
  dataset = dataset.shuffle(buffer_size=buffer_size_,reshuffle_each_iteration=True).repeat(count=None)  #Shuffle to mix up the order of data training per epoch
  dataset = dataset.map(read_image_and_label)   #Mapping the dataset to a tensor
  dataset = dataset.batch(batch_size_)  #Batching the dataset for training
  dataset = dataset.prefetch(None)
  return dataset

def test_data(batch_size_,datasettype):
  #test if the data is what is expected
  dataset=input_fn(batch_size_,datasettype)
  it=dataset.make_one_shot_iterator()
  next_element=it.get_next()
  for i in range(batch_size_):
    image,label=next_element
    image=image[i,:,:,:]*255
    image=tf.cast(image, tf.int32)
    label=label[i,:,:,:]
    #Changing color values of the classes so they allign with the cityscape labe palette
    palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]]
    label=tf.gather(palette, label)
    label = tf.squeeze(label)
  plt.imshow(image)
  plt.show()
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
    inter = tf.to_int32(true_labels & pred_labels)
    union = tf.to_int32(true_labels | pred_labels)
    legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
    ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
    iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return iou

def sparse_Mean_IOU_formean(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(y_true[:,:,0], i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

  #Calculating IOUs for every class
def fn0(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,0)	
def fn1(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,1)	
def fn2(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,2)	
def fn3(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,3)	
def fn4(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,4)	
def fn5(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,5)	
def fn6(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,6)	
def fn7(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,7)	
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

def val_fn0(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,0)	
def val_fn1(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,1)	
def val_fn2(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,2)	
def val_fn3(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,3)	
def val_fn4(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,4)	
def val_fn5(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,5)	
def val_fn6(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,6)	
def val_fn7(y_true, y_pred):return sparse_Mean_IOU(y_true,y_pred,7)	
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
  #Get training dataset
  dataset=input_fn(batch_size_,datasettype = 'training')
  #Get Validation dataset
  datasetval = input_fn(batch_size_,datasettype = 'validation')
  #Defining the resulting metrics
  metrics =["accuracy",fn0,fn1,fn2,fn3,fn4,fn5,fn6,fn7,fn8,fn9,fn10,fn11, fn12, fn13, fn14, fn15, fn16,fn17,fn18,fn19]
  #Restoring model possibility
  if restore == True:
      model = tf.contrib.saved_model.load_keras_model('./saved_models/1561026721')
  else:
      #The Model with Resnet50, BatchNorm, SGD optimizer and Categorical Cross-entropy Loss
      model=tf.keras.models.Sequential()
      inputs=tf.keras.layers.Input(shape=(256,512,3))
      model.add(ResNet50(include_top=False, weights='imagenet',input_tensor=inputs, pooling=None, classes = 20))
      model.add(tf.keras.layers.Conv2D(20,(3,3)))
      model.add(layers.BatchNormalization())   
      model.add(tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x,(256,512))))
      model.summary()
  learning_rate = 0.002
  sgd = tf.keras.optimizers.SGD(learning_rate, momentum=0.8, nesterov=True)
  model.compile(optimizer=sgd,   #SGD with momemtum usually gives good enough results without too many parameters as is the case for ADAMoptimizer
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True),  
                metrics=metrics)  
  history=model.fit(dataset,validation_data = datasetval, epochs=5, steps_per_epoch=1,validation_steps=1) # 2975 Training Images - 1525 Test Images - 500 validation Images
  #Saving the model after training
  saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models/")
  print(history.history.keys())
  
  return model

# Testing the model
def test_model(batch_size_,restore):
  #test if the output is what is expected; prints model generated images of the last batch
  dataset=input_fn(batch_size_,datasettype = 'train')
  datasetval = input_fn(batch_size_,datasettype = 'validation')
  #model=TheModel(batch_size_,restore)
  model = tf.contrib.saved_model.load_keras_model('./saved_models/1561026721')
  it=dataset.make_one_shot_iterator()
  mean_iou_cat=0
  iou_cat = 0
  mean_iou_c = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
  for i in range(batch_size_):
    next_element=it.get_next()
    image_original,label_original=next_element
    output = model.predict_on_batch(image_original)
    pred = output[i,:,:,:]
    #Argmax to find the class with highest probability
    pred=tf.keras.backend.argmax(pred,axis=-1)
    palette= [[0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],[153,153,153],[250,170,30],[220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],[220, 20, 60],[255,  0,  0],[  0,  0,142],[  0,  0, 70],[  0, 60,100],[  0, 80,100],[  0,  0,230],[119, 11, 32]]
    pred=tf.gather(palette, pred)   
    image_original = image_original[i,:,:,:]*255
    image_original=tf.cast(image_original, tf.int32)
    label=label_original[i,:,:,:]
    label=tf.gather(palette, label)
    label=tf.squeeze(label)
    iou=np.empty(0)
    for j in range(20):
      iou_cat=sparse_Mean_IOU(label_original,output,j)
      
      if iou_cat.shape == (0,1):
          iou_cat = 0
      iou=np.append(iou,iou_cat) 
      iou=iou.flatten()
    mean_iou_c= mean_iou_c+ iou
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
  plt.imshow(pred)
  plt.show()
  plt.imshow(image_original)
  plt.show()
  plt.imshow(label)
  plt.show()

#Main Function
if __name__=='__main__':
  #Choosing between loading or saving the model
  restore = True  
  batch_size_ = 40
  #test_data(batch_size_,datasettype = 'augment')
  #TheModel(batch_size_,restore)
  test_model(batch_size_,restore)


