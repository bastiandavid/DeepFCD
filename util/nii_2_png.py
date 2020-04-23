#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 12:02:04 2018

@author: bdavid
"""

import os
import sys, getopt
import numpy as np
import nibabel as nib
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageStat
import warnings
warnings.filterwarnings("ignore")

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
def image_padding(img, outsize):
    """ Function to pad and scale image to desired size"""
    
    old_size = img.size  # old_size[0] is in (width, height) format
    ratio = float(outsize)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image
    # thumbnail is a in-place operation
    # im.thumbnail(new_size, Image.ANTIALIAS)
    img = img.resize(new_size, resample=Image.BICUBIC)
    # create a new image and paste the resized on it
    new_img = Image.new("L", (outsize, outsize))
    new_img.paste(img, ((outsize-new_size[0])//2,
                    (outsize-new_size[1])//2))
    
    return new_img
        
def save_to_png(filepath, prefix, outtype, outsize, outdir, cutoff):
    """Function to save NIFTI slicewise as png with right scaling
    and in radiological convention (left is right, right is left)"""
    img=nib.load(filepath)
    data=img.get_data()
    
    if not os.path.exists(filepath):
        print 'Filepath "'+filepath+'" does not exist. Exiting'
        sys.exit(2)
    
    if not os.path.exists(outdir): os.makedirs(outdir)
    
    print "Producing images for prefix: "+prefix    
    for i in range(np.size(data,1)):
        imgname=outdir+"/"+prefix+"_slice"+str("%03d" % (i,))+"."+outtype.lower()
        sliceimg=scipy.misc.toimage(np.fliplr(np.flipud(data[:,i,:].T)), cmin=0.0, cmax=data.max())
        sliceimg=image_padding(sliceimg, outsize)
        if ImageStat.Stat(sliceimg).mean[0] >= cutoff: sliceimg.save(imgname,outtype.upper())

def main(argv):
    
    filepath=''
    prefix=''
    outtype=''
    outdir=''
    
    try:
        opts, args = getopt.getopt(argv,"hi:p:t:s:o:c:",["infile=","prefix=","output_type=","outsize=","outdir=","cutoff="])
    except getopt.GetoptError:
      print 'nii_2_png.py -i <inputfile_path> -p <prefix_output> -t <output_type (e.g. PNG)> -s <output_size> -o <output_directory> -c <intensity cutoff>'
      sys.exit(2)
      
    for opt, arg in opts:
        if opt == '-h':
            print 'nii_2_png.py -i <inputfile_path> -p <prefix_output> -t <output_type (e.g. PNG)> -s <output_size> -o <output_directory> -c <intensity cutoff>'
            sys.exit()
        elif opt in ("-i", "--infile"):
            filepath = arg
        elif opt in ("-p", "--prefix"):
            prefix = arg
        elif opt in ("-t","--output_type"):
            outtype = arg
        elif opt in ("-s","--outsize"):
            outsize = int(arg)
        elif opt in ("-o","--outdir"):
            outdir = arg
        elif opt in ("-c","--cutoff"):
            cutoff = int(arg)

    save_to_png(filepath, prefix, outtype, outsize, outdir, cutoff)
    
main(sys.argv[1:])

        