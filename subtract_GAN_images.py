#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:13:02 2018

@author: bdavid
"""

import os
import argparse
import glob
#import re
from PIL import Image
from PIL import ImageChops
import warnings
warnings.filterwarnings("ignore")

def subtract_GAN_images(realdir, fakedir, subjid, direction, pngoutdir):
    
    fakeslices=sorted(glob.glob(os.path.join(fakedir,'*.png')))
    realslices=sorted(glob.glob(os.path.join(realdir,'*.png')))
    
    for fake_png, real_png in zip(fakeslices, realslices) :
        
        fakeslice=Image.open(fake_png)
        realslice=Image.open(real_png)
        
        if direction == 'real-fake':
            outslice=ImageChops.subtract(realslice,fakeslice)
        elif direction == 'fake-real':
            outslice=ImageChops.subtract(fakeslice,realslice)
            
        if not os.path.exists(pngoutdir):
            os.makedirs(pngoutdir)
            
        outslice.save(os.path.join(pngoutdir,fake_png.split('/')[-1]))
        
        
parser = argparse.ArgumentParser(description='Creates difference images of synthetic and real modalities. Output will be PNGs.')
parser.add_argument("-rd", "--realdir", help="path to directory of real modality PNGs")
parser.add_argument("-fd", "--fakedir", help="path to directory of fake modality PNGs")
parser.add_argument("-s", "--subjid", help="subject ID, used also as prefix for output files")
parser.add_argument("-dir", "--directionality", choices=['real-fake','fake-real'], help="directionality of subtraction")
parser.add_argument("-od", "--outdir", help="path to PNG output directory. Creates directory if it doesn't exist")

args=parser.parse_args()

subtract_GAN_images(args.realdir, args.fakedir, args.subjid, args.directionality, args.outdir)



        
    