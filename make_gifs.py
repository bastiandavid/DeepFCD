#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:21:47 2020

@author: bdavid
"""


import os
import numpy as np
import glob
import imageio
from PIL import Image

INPUTPATH="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/T1/test"
TARGETPATH="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/FLAIR/test"
SYNTHPATH="/home/bdavid/Deep_Learning/playground/intensity_rescaled/test"
OUTPATH="/home/bdavid/Deep_Learning/DeepFCD/assets/example_outputs"



def horizontal_concat(input_img, target_img, synth_img):
    images = [Image.open(x) for x in [input_img, target_img, synth_img]]
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('L', (total_width, max_height))
    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    
    return new_im

input_list = sorted(glob.glob(os.path.join(INPUTPATH,"11362*.png")))[174::-1]
target_list = sorted(glob.glob(os.path.join(TARGETPATH,"11362*.png")))[174::-1]
synth_list = sorted(glob.glob(os.path.join(SYNTHPATH,"11362*.png")))[174::-1]

gif_images = []

for input_img, target_img, synth_img in zip(input_list, target_list, synth_list):
    gif_images.append(imageio.imread(horizontal_concat(input_img, target_img, synth_img)))

input_list=input_list[::-1]
target_list=target_list[::-1]
synth_list=synth_list[::-1]

for input_img, target_img, synth_img in zip(input_list, target_list, synth_list):
    gif_images.append(imageio.imread(horizontal_concat(input_img, target_img, synth_img)))

imageio.mimsave(os.path.join(OUTPATH,"synth_flair_1.gif"),gif_images, duration=0.08)