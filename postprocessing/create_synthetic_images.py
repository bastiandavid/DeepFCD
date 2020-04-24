#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:06:35 2020

Script to apply the trained generator network (U-net) to create synthetic MR-images.
Images will be saved as PNGs. Images in RAW_OUTPATH are saved without post-hoc intensity
scaling. I recommend using the images in OUTPATH instead, after histogram matching and
intensity scaling for best results.
IMPORTANT: Saving and loading the keras model with CPU at the moment only works
with the tf.nightly build! GPU version should also work with the stable 2.1.0 version

@author: bdavid
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np
import glob
from PIL import Image
from IPython import display
from skimage.exposure import match_histograms

MODEL = '../models/FLAIR_2_T1_cor/generator'

TARGETPATH = '/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/T1/'
INPUTPATH ='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/FLAIR/'
TARGET_PADDING_PATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/T1_paddings/'
INPUT_PADDING_PATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/FLAIR_paddings/'

INFILES='test/*.png'
RAW_OUTPATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/raw_synth_T1/'
OUTPATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/synth_T1/'

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
INPUT_CHANNELS = 7
if INPUT_CHANNELS % 2 == 0:
    print('Even no. of slices not supported, setting INPUT_CHANNELS to ',INPUT_CHANNELS+1)
    INPUT_CHANNELS += 1
    
def file_exists(image_file,slicenum,slice_of_interest):
    return tf.io.gfile.exists(tf.strings.regex_replace(image_file,slicenum,str(slice_of_interest.numpy()).zfill(3)+'.').numpy())

def load_padding(padding_path,subjid,first):
    if first:
        return tf.io.read_file(padding_path+subjid+'_first_mean_padding.png')
    else:
        return tf.io.read_file(padding_path+subjid+'_last_mean_padding.png')
    
def load_curr_slice(image_file,slicenum,curr_slice):
    return tf.io.read_file(tf.strings.regex_replace(image_file,slicenum,str(curr_slice.numpy()).zfill(3)+'.'))

def load(image_file):

  real_image_file=tf.strings.regex_replace(image_file,INPUTPATH,TARGETPATH)
  #create 3D image stack for multi-channel input with mean padding
  #input_imagelist,real_imagelist = tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS), tf.float32), tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS), tf.float32)    
  input_imagelist = tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS), tf.float32)
  slicenum="([0-9]{3})\."
  subjid=tf.strings.split(tf.strings.split(image_file,sep='/')[-1],sep='_')[0]
  mid_slice=int(tf.strings.substr(image_file, -7, 3))
  halfstack = lo_idx = hi_idx = INPUT_CHANNELS//2
  min_slice=mid_slice-lo_idx
  max_slice=mid_slice+hi_idx
  num_curr_slice=0


  while not tf.py_function(file_exists,[image_file, slicenum, min_slice],Tout=tf.bool):
    lo_idx -= 1
    min_slice +=1

  while not tf.py_function(file_exists,[image_file, slicenum, max_slice],Tout=tf.bool):
    hi_idx -=1
    max_slice -=1

  if halfstack-lo_idx != 0:
    
    #input_image = tf.io.read_file(INPUT_PADDING_PATH+subjid+'_first_mean_padding.png')
    input_image = tf.py_function(load_padding, [INPUT_PADDING_PATH,subjid,True],Tout=tf.string)
    input_image = tf.image.decode_png(input_image,channels=1)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32) 
    
#     real_image = tf.py_function(load_padding, [TARGET_PADDING_PATH,subjid,True],Tout=tf.string)
#     real_image = tf.image.decode_png(real_image,channels=1)
#     real_image = tf.image.convert_image_dtype(real_image, tf.float32) 
    
    for i in range(halfstack-lo_idx): 
      input_imagelist=tf.concat([input_imagelist[...,:num_curr_slice], 
                                 input_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
                                                       tf.float32)], axis=2)
#       real_imagelist=tf.concat([real_imagelist[...,:num_curr_slice], 
#                                 real_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
#                                                      tf.float32)], axis=2)
      num_curr_slice += 1
    
      input_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
#       real_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
    
  for curr_slice in range(min_slice, max_slice+1):
    
    #input_image = tf.io.read_file(tf.strings.regex_replace(image_file,slicenum,str(curr_slice).zfill(3)+'.'))
    input_image = tf.py_function(load_curr_slice, 
                                 [image_file,slicenum,curr_slice], 
                                 Tout=tf.string)
    input_image = tf.image.decode_png(input_image,channels=1)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32) 
    
    #real_image = tf.io.read_file(tf.strings.regex_replace(real_image_file,slicenum,str(curr_slice).zfill(3)+'.'))
#     real_image = tf.py_function(load_curr_slice, 
#                                 [real_image_file,slicenum,curr_slice], 
#                                 Tout=tf.string)
#     real_image = tf.image.decode_png(real_image,channels=1)
#     real_image = tf.image.convert_image_dtype(real_image, tf.float32) 
    
    
    input_imagelist=tf.concat([input_imagelist[...,:num_curr_slice], 
                               input_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
                                                     tf.float32)], axis=2)
#     real_imagelist=tf.concat([real_imagelist[...,:num_curr_slice], 
#                               real_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
#                                                    tf.float32)], axis=2)
    num_curr_slice += 1
    
    input_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
#     real_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])    
                                      
  if halfstack-hi_idx != 0:
    
    input_image = tf.py_function(load_padding, [INPUT_PADDING_PATH,subjid,False],Tout=tf.string)
    input_image = tf.image.decode_png(input_image,channels=1)
    input_image = tf.image.convert_image_dtype(input_image, tf.float32) 

    
#     real_image = tf.py_function(load_padding, [TARGET_PADDING_PATH,subjid,False],Tout=tf.string)
#     real_image = tf.image.decode_png(real_image,channels=1)
#     real_image = tf.image.convert_image_dtype(real_image, tf.float32) 
    
    for i in range(halfstack-hi_idx): 
      input_imagelist=tf.concat([input_imagelist[...,:num_curr_slice], 
                                 input_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
                                                       tf.float32)], axis=2)
#       real_imagelist=tf.concat([real_imagelist[...,:num_curr_slice], 
#                                 real_image, tf.zeros((IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS-num_curr_slice-1), 
#                                                      tf.float32)], axis=2)
      num_curr_slice += 1
      input_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
#       real_imagelist.set_shape([IMG_WIDTH, IMG_HEIGHT, INPUT_CHANNELS])
      
  real_image = tf.io.read_file(real_image_file)
  real_image = tf.image.decode_png(real_image, channels=1)
  real_image = tf.image.convert_image_dtype(real_image, tf.float32)
  
#   return input_imagelist, real_imagelist
  return input_imagelist, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def intensity_rescale(synth_img, real_img):
    
    real_img=np.array(Image.open(real_img))
    synth_img=np.array(Image.open(synth_img))

    min_real=np.min(real_img)
    max_real=np.max(real_img)
    
    scale=(max_real-min_real)/(np.max(synth_img)-np.min(synth_img))
    offset=max_real - scale*np.max(synth_img)
    
    synth_img_scaled=scale*synth_img+offset
   
    return Image.fromarray(np.uint8(synth_img_scaled))

def histo_matching(synth_img, real_img):
    
    real_img=np.array(Image.open(real_img))
    synth_img=np.array(Image.open(synth_img))
    
    synth_img_scaled=match_histograms(synth_img,real_img)
    
    return Image.fromarray(np.uint8(synth_img_scaled))

test_dataset = tf.data.Dataset.list_files(INPUTPATH+INFILES, shuffle=False)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = tf.keras.models.load_model(MODEL)

os.makedirs(os.path.join(RAW_OUTPATH,'test'), exist_ok=True)

# Run the trained model on a few examples from the test dataset
for (inp, tar), path in zip(test_dataset,
                            tf.data.Dataset.list_files(INPUTPATH+INFILES,shuffle=False)):
    
  prediction = generator(inp, training=True)

  outfile = tf.strings.regex_replace(path,INPUTPATH,RAW_OUTPATH)
  display.clear_output(wait=True)
  print('writing '+outfile.numpy().decode("utf-8").split('/')[-1])
  tf.keras.preprocessing.image.save_img(outfile.numpy(), prediction[0], file_format='png')

  
raw_synth_list=glob.glob(os.path.join(RAW_OUTPATH,INFILES))
real_list=glob.glob(os.path.join(TARGETPATH,INFILES))

os.makedirs(os.path.join(OUTPATH,'test'), exist_ok=True)

for synth_img, real_img in zip(raw_synth_list, real_list):
    
    # MinMax Intensity scaling (not recommended):
    # synth_img_minmax_scaled= intensity_rescale(synth_img, real_img)
    # synth_img_minmax_scaled.save(os.path.join(OUTPATH,'test','minmax',synth_img.split('/')[-1]))
    
    display.clear_output(wait=True)
    print('writing '+synth_img.split('/')[-1])
    synth_img_histo_scaled = histo_matching(synth_img, real_img)
    synth_img_histo_scaled.save(os.path.join(OUTPATH,'test',synth_img.split('/')[-1]))