#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:32:39 2018

@author: bdavid
"""

import sys

from scipy.misc import imread
from numpy.linalg import norm
from scipy import sum, average
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim as ssim

def main():
    file1, file2, file3 = sys.argv[1:1+3]
    # read images as 2D arrays (convert to grayscale for simplicity)
    img1 = np.array(Image.open(file1).convert('L'))
    img2 = np.array(Image.open(file2).convert('L'))
    mask = np.array(Image.open(file3).convert('L'))
    
    img1 = img1 * mask
    img2 = img2 * mask
    # compare
    n_l1, n_l2 , ssim = compare_images(img1, img2)
    print n_l1,"\t", n_l2, "\t", ssim
    
def compare_images(img1, img2):
    # normalize to compensate for exposure difference
    #img1 = normalize(img1)
    #img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    l1_norm = sum(abs(diff))  # Manhattan norm
    l2_norm = sum(diff**2)/np.size(img1)
    structural_sim = ssim(img1, img2)
    
    return (l1_norm, l2_norm, structural_sim)

if __name__ == "__main__":
    main()