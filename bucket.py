from google.cloud import storage
import tensorflow as tf
import numpy as np
import os

#Get the json from your bucket to get access
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='DeepLearningSegmentation-183071a6a2da.json'


#Create client and bucket
client = storage.Client()
bucket = client.get_bucket("deeplearning_frouke_jesse")

trainlength = sum(1 for line in open('names_images_train.txt'))
trainlines_img = open('names_images_train.txt').readlines()
trainlines_lab = open('names_labels_train.txt').readlines()

vallength = sum(1 for line in open('names_images_val.txt'))
vallines_img = open('names_images_val.txt').readlines()
vallines_lab = open('names_labels_val.txt').readlines()
#Put this in for loop to iterate over .txt file 'name'
for i in range(trainlength):
    name = trainlines_img[i]
    get_image_train = bucket.blob('gtFine/val/leftImg8bit/train/*/'+name+'.png') #this is the directory from the bucket
    train_img = get_image_train.download_to_filename('/home/kaasbomber/images/train/image/'+name+'.png') # this is the directory and filename you want
for j in range(trainlength):
    name = trainlines_lab[j]
    get_label_train = bucket.blob('gtFine/val/gtFine/train/*/'+name+'.png')
    get_label_train.download_to_filename('/home/kaasbomber/images/train/label/'+name+'.png') # this is the directory and filename you want

for m in range(vallength):
    name = vallines_img[m]
    get_image_val = bucket.blob('gtFine/val/leftImg8bit/train/*/'+name+'.png') #this is the directory from the bucket
    val_img = get_image_val.download_to_filename('/home/kaasbomber/images/val/image/'+name+'.png') # this is the directory and filename you want
for n in range(vallength):
    name = vallines_lab[n]
    get_label_val = bucket.blob('gtFine/val/gtFine/train/*/'+name+'.png')
    get_label_val.download_to_filename('/home/kaasbomber/images/val/label/'+name+'.png') # this is the directory and filename you want


# =============================================================================
# #Put this in for loop to iterate over .txt file 'name'
# get_image = bucket.blob('images/train/xxxxxx.png') #this is the directory from the bucket
# train_img = get_image.download_to_filename('/home/youname/images/train/'+name+'.png') # this is the directory and filename you want
# 
# =============================================================================
