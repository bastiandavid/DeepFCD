#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:57:22 2020

Script to create specific cross-validation configurations for DeepMedic.
Needs to be refactored to be more dynamic. At the moment only implemented
for 7-channel input.

@author: bdavid
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
import os

lesions = pd.read_csv('/home/bdavid/Deep_Learning/data/bonn/administration/lists/lesion_volumes.txt',
                      delim_whitespace=True)

session='7ch_FCD_crossval'
random_state=42
n_splits=5
n_repeats=3
outpath=os.path.join('/home/bdavid/Deep_Learning/deepmedic/examples/configFiles/',session)
inpath='/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/nii/deepmedic_input'

rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

repeat=1
idx=1
split = 1
for train, test in rkf.split(lesions['ID']): 
    
    curr_path=str(repeat)+'_'+str(split)
    filename_train='train_'+str(repeat)+'_'+str(split)+'.txt'
    filename_test='test_'+str(repeat)+'_'+str(split)+'.txt'
    
    curr_train=lesions['ID'].iloc[train].values
    curr_test=lesions['ID'].iloc[test].values
    
    os.makedirs(os.path.join(outpath,'train',curr_path), exist_ok=True)
    os.makedirs(os.path.join(outpath,'test',curr_path), exist_ok=True)
    np.savetxt(os.path.join(outpath,'train',curr_path,filename_train),curr_train,delimiter='\n',fmt='%d')
    np.savetxt(os.path.join(outpath,'test',curr_path,filename_test),curr_test,delimiter='\n',fmt='%d')
    
    for sbj in curr_train:
        
        with open(os.path.join(outpath,'train',curr_path,'T1.txt'),"a") as T1:
            print(os.path.join(inpath,str(sbj)+'_T1.nii.gz'), file=T1)
        with open(os.path.join(outpath,'train',curr_path,'FLAIR.txt'),"a") as FLAIR:
            print(os.path.join(inpath,str(sbj)+'_FLAIR.nii.gz'), file=FLAIR)
        with open(os.path.join(outpath,'train',curr_path,'diff.txt'),"a") as diff:
            print(os.path.join(inpath,str(sbj)+'_diff.nii.gz'), file=diff)
        with open(os.path.join(outpath,'train',curr_path,'weights.txt'),"a") as weights:
            print(os.path.join(inpath,str(sbj)+'_weights.nii.gz'), file=weights)
        with open(os.path.join(outpath,'train',curr_path,'extension.txt'),"a") as extension:
            print(os.path.join(inpath,str(sbj)+'_extension.nii.gz'), file=extension)
        with open(os.path.join(outpath,'train',curr_path,'junction.txt'),"a") as junction:
            print(os.path.join(inpath,str(sbj)+'_junction.nii.gz'), file=junction)
        with open(os.path.join(outpath,'train',curr_path,'thickness.txt'),"a") as thickness:
            print(os.path.join(inpath,str(sbj)+'_thickness.nii.gz'), file=thickness)
        with open(os.path.join(outpath,'train',curr_path,'roi.txt'),"a") as roi:
            print(os.path.join(inpath,str(sbj)+'_roi.nii.gz'), file=roi)
        with open(os.path.join(outpath,'train',curr_path,'mask.txt'),"a") as mask:
            print(os.path.join(inpath,str(sbj)+'_mask.nii.gz'), file=mask)
            
    for sbj in curr_test:
        
        with open(os.path.join(outpath,'test',curr_path,'T1.txt'),"a") as T1:
            print(os.path.join(inpath,str(sbj)+'_T1.nii.gz'), file=T1)
        with open(os.path.join(outpath,'test',curr_path,'FLAIR.txt'),"a") as FLAIR:
            print(os.path.join(inpath,str(sbj)+'_FLAIR.nii.gz'), file=FLAIR)
        with open(os.path.join(outpath,'test',curr_path,'diff.txt'),"a") as diff:
            print(os.path.join(inpath,str(sbj)+'_diff.nii.gz'), file=diff)
        with open(os.path.join(outpath,'test',curr_path,'weights.txt'),"a") as weights:
            print(os.path.join(inpath,str(sbj)+'_weights.nii.gz'), file=weights)
        with open(os.path.join(outpath,'test',curr_path,'extension.txt'),"a") as extension:
            print(os.path.join(inpath,str(sbj)+'_extension.nii.gz'), file=extension)
        with open(os.path.join(outpath,'test',curr_path,'junction.txt'),"a") as junction:
            print(os.path.join(inpath,str(sbj)+'_junction.nii.gz'), file=junction)
        with open(os.path.join(outpath,'test',curr_path,'thickness.txt'),"a") as thickness:
            print(os.path.join(inpath,str(sbj)+'_thickness.nii.gz'), file=thickness)
        with open(os.path.join(outpath,'test',curr_path,'roi.txt'),"a") as roi:
            print(os.path.join(inpath,str(sbj)+'_roi.nii.gz'), file=roi)
        with open(os.path.join(outpath,'test',curr_path,'mask.txt'),"a") as mask:
            print(os.path.join(inpath,str(sbj)+'_mask.nii.gz'), file=mask)
        with open(os.path.join(outpath,'test',curr_path,'prediction_names.txt'),"a") as pred:
            print(os.path.join(inpath,str(sbj)+'_prediction.nii.gz'), file=pred)
            
    with open(os.path.join(outpath,'train','trainConfig.cfg'), 'r') as generic_train:
        filedata = generic_train.read()
    filedata = filedata.replace('session_name',session+'_'+str(repeat)+'_'+str(split))
    with open(os.path.join(outpath,'train',curr_path,'train_'+str(repeat)+'_'+str(split)+'.cfg'), 'w') as curr_config:
        curr_config.write(filedata)
        
    with open(os.path.join(outpath,'test','testConfig.cfg'), 'r') as generic_test:
        filedata = generic_test.read()
    filedata = filedata.replace('session_name',session+'_'+str(repeat)+'_'+str(split))
    with open(os.path.join(outpath,'test',curr_path,'test_'+str(repeat)+'_'+str(split)+'.cfg'), 'w') as curr_config:
        curr_config.write(filedata)
            
        
    split += 1
    if idx%n_splits == 0: 
        repeat += 1
        split = 1
    idx += 1