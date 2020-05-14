#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:46:28 2020

@author: bdavid
"""



from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import numpy as np
import glob
from PIL import Image, ImageChops
from skimage.exposure import match_histograms
import nibabel as nib
from tqdm import tqdm

# -------- USER INPUT ----------

GENERATOR = '../models/T1_2_FLAIR_cor/generator'
DISCRIMINATOR = '../models/T1_2_FLAIR_cor/discriminator'
DIRECTION = 'real-fake'
INPUT_MODALITY = 'T1'
TARGET_MODALITY = 'FLAIR'
DATAPATH = '/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/png'
NIIPATH = '/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/nii'
SUBJID = '' # just single subject? if none given, all files are processed
CREATE_NIFTI = True # create niftis for input, synthetic and diff images?
DATASET = '' # test or train? just some directory prefix

#-------------------------------


INFILES= SUBJID+'*.png' 
TARGETPATH = os.path.join(DATAPATH,TARGET_MODALITY,DATASET,'')
INPUTPATH = os.path.join(DATAPATH,INPUT_MODALITY,DATASET,'')
TARGET_PADDING_PATH= os.path.join(DATAPATH,TARGET_MODALITY+'_paddings',DATASET,'')
INPUT_PADDING_PATH=os.path.join(DATAPATH,INPUT_MODALITY+'_paddings',DATASET,'')

RAW_OUTPATH=os.path.join(DATAPATH,'disc_output_'+TARGET_MODALITY,DATASET,'')
if DIRECTION == 'real-fake':
    DIFF_OUTPATH=os.path.join(DATAPATH,'diff_'+'real_'+TARGET_MODALITY+
                         '-'+'synth_'+TARGET_MODALITY,DATASET,'')
    DIFF_NII=os.path.join(NIIPATH,'diff_'+'real_'+TARGET_MODALITY+
                         '-'+'synth_'+TARGET_MODALITY,DATASET,'')
else:
    DIFF_OUTPATH=os.path.join(DATAPATH,'diff_'+'synth_'+TARGET_MODALITY+
                         '-'+'real_'+TARGET_MODALITY,DATASET,'')
    DIFF_NII=os.path.join(NIIPATH,'diff_'+'synth_'+TARGET_MODALITY+
                         '-'+'real_'+TARGET_MODALITY,DATASET,'')
OUTPATH=os.path.join(DATAPATH,'synth_'+TARGET_MODALITY,DATASET,'')

TARGET_NII = os.path.join(NIIPATH,TARGET_MODALITY,DATASET,'')
INPUT_NII = os.path.join(NIIPATH,INPUT_MODALITY,DATASET,'')
SYNTH_NII = os.path.join(NIIPATH,'synth_'+TARGET_MODALITY,DATASET,'')
GAN_INPUT_NII = os.path.join(NIIPATH,'gan_input_'+INPUT_MODALITY,DATASET,'')

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

def subtract_images(synth_img, real_img, direction=DIRECTION):
    
 
#    synth_img=Image.open(synth_img)
    real_img=Image.open(real_img)
    
    if direction == 'real-fake':
        out_img=ImageChops.subtract(real_img,synth_img)
    elif direction == 'fake-real':
        out_img=ImageChops.subtract(synth_img,real_img)
        
    return out_img

def to_nifti(subjid, realnii, inputdir, outname):
    
    real_nifti=nib.load(realnii)

    first_slice=True
    for png in sorted(glob.glob(os.path.join(inputdir, subjid+'*.png'))):
        
        curr_slice=np.array(Image.open(png).convert('L'))
        #curr_slice=np.fliplr(np.flipud(curr_slice))
        
        if first_slice:
            vol_array=curr_slice
            first_slice=False
        else:
            vol_array=np.dstack((vol_array, curr_slice))
            
    final_nifti=nib.Nifti1Image(np.rot90(np.rot90(vol_array),axes=(2,1)), real_nifti.affine, header=real_nifti.header)
    final_nifti.to_filename(outname)


test_dataset = tf.data.Dataset.list_files(INPUTPATH+INFILES, shuffle=False)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = tf.keras.models.load_model(GENERATOR)
discriminator = tf.keras.models.load_model(DISCRIMINATOR)


os.makedirs(os.path.join(RAW_OUTPATH), exist_ok=True)

# Run the trained model on a few examples from the test dataset
print()
for (inp, tar), path, i in zip(test_dataset,
                            tf.data.Dataset.list_files(INPUTPATH+INFILES,shuffle=False),
                            tqdm(range(len(glob.glob(os.path.join(TARGETPATH,INFILES)))),
                                 desc='Creating raw synthetic images')):
    
  prediction = generator(inp, training=True)
  
  disc_output = discriminator([inp[...,INPUT_CHANNELS//2,tf.newaxis], prediction], 
                                          training=True)

  outfile = tf.strings.regex_replace(path,INPUTPATH,RAW_OUTPATH)
  #print('\rwriting '+outfile.numpy().decode("utf-8").split('/')[-1], end='')
  tf.keras.preprocessing.image.save_img(outfile.numpy(), disc_output[0], file_format='png')