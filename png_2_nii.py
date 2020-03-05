#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:44:15 2018

@author: bdavid
"""

import os
import argparse
import numpy as np
import nibabel as nib
import glob
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def save_to_nii(inputdir, realnii, outname):
     
    real_nifti=nib.load(realnii)

    first_slice=True
    for png in sorted(glob.glob(os.path.join(inputdir,'*.png'))):
        
        curr_slice=np.array(Image.open(png).convert('L'))
        #curr_slice=np.fliplr(np.flipud(curr_slice))
        
        if first_slice:
            vol_array=curr_slice
            first_slice=False
        else:
            vol_array=np.dstack((vol_array, curr_slice))
            
    final_nifti=nib.Nifti1Image(np.rot90(np.rot90(vol_array),axes=(2,1)), real_nifti.affine, header=real_nifti.header)
    final_nifti.to_filename(outname)
    
    
    
parser = argparse.ArgumentParser(description='Creates NIFTI images out of PNGs.')
parser.add_argument("-id", "--inputdir", help="path to input directory for PNGs")
parser.add_argument("-rn", "--realnii", help="real nifti input to copy header and affine information from")
parser.add_argument("-out", "--outname", help="output name for nifti")

args=parser.parse_args()

save_to_nii(args.inputdir, args.realnii, args.outname)
        