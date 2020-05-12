#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:06:53 2020

@author: bdavid
"""

import os
import re
import glob
import imageio
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image

INPUT_CHANNELS = 7
if INPUT_CHANNELS % 2 == 0:
    print('Even no. of slices not supported, setting INPUT_CHANNELS to ',INPUT_CHANNELS+1)
    INPUT_CHANNELS += 1

OUTPATH='/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/png'
INPATH='/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/png'
modalities = ['FLAIR','T1']

#folders=['test','train']
folders=['']
subjid=re.compile('([0-9]+)_')
for modality in modalities:
    
    os.makedirs(os.path.join(OUTPATH,modality+'_paddings'))
    
    for folder in folders:
        
        path_to_images=os.path.join(INPATH,modality,folder)
        ids=set()
        for f in os.listdir(path_to_images):
            ids.add(subjid.findall(f)[0])
        
        
        for subject in ids:
            first_slices=sorted(glob.glob(os.path.join(path_to_images,subject)+'*'))[0:INPUT_CHANNELS]
            last_slices=sorted(glob.glob(os.path.join(path_to_images,subject)+'*'))[-INPUT_CHANNELS:]
            
            first_mean=np.average(np.array([imageio.imread(im) for im in first_slices]),axis=0)
            last_mean=np.average(np.array([imageio.imread(im) for im in last_slices]),axis=0)
            
            first_padding=Image.fromarray(first_mean.astype('uint8'))
            last_padding=Image.fromarray(last_mean.astype('uint8'))
            
            first_padding.save(os.path.join(OUTPATH,modality+'_paddings',subject+'_first_mean_padding.png'))
            last_padding.save(os.path.join(OUTPATH,modality+'_paddings',subject+'_last_mean_padding.png'))
            