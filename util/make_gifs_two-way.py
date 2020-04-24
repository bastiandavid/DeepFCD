#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:21:47 2020

@author: bdavid
"""


import os
import glob
import imageio
import io
from PIL import Image, ImageDraw, ImageFont

INPUTPATH="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/FLAIR/test"
TARGETPATH="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/T1/test"
SYNTHPATH1="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/synth_FLAIR/test"
SYNTHPATH2="/home/bdavid/Deep_Learning/playground/fake_flair_2d/png_cor/synth_T1/test"
OUTPATH="/home/bdavid/Deep_Learning/DeepFCD/assets/example_outputs"

subjid="555-nase"
#set offset to not show face in GIFs
offset=190



def horizontal_concat(input_img, target_img, synth_img_1, synth_img_2, labels=True):
    
    if labels:
        label_img=createLabelImg(Image.open(input_img))
        images = [Image.open(x) for x in [input_img, target_img, synth_img_1, synth_img_2]]
        images.insert(2,label_img)
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('L', (total_width, max_height))
        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]
          
        imgByteArr = io.BytesIO()
        new_im.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        
        return imgByteArr
    
    else:
        images = [Image.open(x) for x in [input_img, target_img, synth_img_1, synth_img_2]]
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('L', (total_width, max_height))
        
        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]
          
        imgByteArr = io.BytesIO()
        new_im.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        
        return imgByteArr

def createLabelImg(input_img):
    
    label_img = Image.new('L',(input_img.size[0]//2,input_img.size[1]), color = 'black')
    label = ImageDraw.Draw(label_img)
    imgfont = ImageFont.truetype(font='/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf',
                                 size=20)
    label.text((10,50), "synthetic -->", font=imgfont, fill='white')
    label.text((10,200), "<-- real", font=imgfont, fill='white')

    
    return label_img

input_list = sorted(glob.glob(os.path.join(INPUTPATH,subjid+"*.png")))[offset::-1]
target_list = sorted(glob.glob(os.path.join(TARGETPATH,subjid+"*.png")))[offset::-1]
synth_list_1 = sorted(glob.glob(os.path.join(SYNTHPATH1,subjid+"*.png")))[offset::-1]
synth_list_2 = sorted(glob.glob(os.path.join(SYNTHPATH2,subjid+"*.png")))[offset::-1]

gif_images = []

for input_img, target_img, synth_img_1, synth_img_2 in zip(input_list, target_list, 
                                                           synth_list_1, synth_list_2):
    gif_images.append(imageio.imread(horizontal_concat(input_img, target_img, 
                                                       synth_img_1, synth_img_2)))

input_list=input_list[::-1]
target_list=target_list[::-1]
synth_list_1=synth_list_1[::-1]
synth_list_2=synth_list_2[::-1]

for input_img, target_img, synth_img_1, synth_img_2 in zip(input_list, target_list, 
                                                           synth_list_1, synth_list_2):
    gif_images.append(imageio.imread(horizontal_concat(input_img, target_img, 
                                                       synth_img_1, synth_img_2)))

no_of_gifs=len(glob.glob(os.path.join(OUTPATH,"T1_FLAIR_SYNTH-T1_SYNTH-FLAIR*.gif")))

imageio.mimsave(os.path.join(OUTPATH,"T1_FLAIR_SYNTH-T1_SYNTH-FLAIR_"+str(no_of_gifs).zfill(2)+".gif"),gif_images, duration=0.08)

#if compression needed, install pygifsicle and uncomment following lines
from pygifsicle import optimize
optimize(os.path.join(OUTPATH,"T1_FLAIR_SYNTH-T1_SYNTH-FLAIR_"+str(no_of_gifs).zfill(2)+".gif"))