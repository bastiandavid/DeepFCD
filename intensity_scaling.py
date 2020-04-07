#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:24:20 2020

Rescaling exported synthetic PNGs to the intensity range of real PNGs.
MinMax-Scaling not recommended. Better use Histogram Matching
TODO: Integrate directly into apply_GAN notebook 

@author: bdavid
"""

import os
import numpy as np
import glob
from PIL import Image
#from matplotlib import pyplot as plt
from skimage.exposure import match_histograms

SYNTHPATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/synth_FLAIR/'
REALPATH='/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/FLAIR/'
OUTPATH='/home/bdavid/Deep_Learning/playground/intensity_rescaled/'

def intensity_rescale(synth_img, real_img):
    
    real_img=np.array(Image.open(real_img))
    synth_img=np.array(Image.open(synth_img))

    min_real=np.min(real_img)
    max_real=np.max(real_img)
    
    scale=(max_real-min_real)/(np.max(synth_img)-np.min(synth_img))
    offset=max_real - scale*np.max(synth_img)
    
    synth_img_scaled=scale*synth_img+offset
    # plt.figure()
    # plt.imshow(synth_img,cmap='gray',vmin=0,vmax=255)
    # plt.figure()
    # plt.imshow(real_img,cmap='gray',vmin=0,vmax=255)
    # plt.figure()
    # plt.imshow(synth_img_scaled.astype(int),cmap='gray',vmin=0,vmax=255)
    
    return Image.fromarray(np.uint8(synth_img_scaled))

def histo_matching(synth_img, real_img):
    
    real_img=np.array(Image.open(real_img))
    synth_img=np.array(Image.open(synth_img))
    
    synth_img_scaled=match_histograms(synth_img,real_img)
    
    return Image.fromarray(np.uint8(synth_img_scaled))

synth_list=glob.glob(os.path.join(SYNTHPATH,"test/*.png"))
real_list=glob.glob(os.path.join(REALPATH,"test/*.png"))

for synth_img, real_img in zip(synth_list, real_list):
    
    synth_img_minmax_scaled= intensity_rescale(synth_img, real_img)
    synth_img_minmax_scaled.save(os.path.join(OUTPATH,'test','minmax',synth_img.split('/')[-1]))
    
    synth_img_histo_scaled = histo_matching(synth_img, real_img)
    synth_img_histo_scaled.save(os.path.join(OUTPATH,'test','histo',synth_img.split('/')[-1]))
