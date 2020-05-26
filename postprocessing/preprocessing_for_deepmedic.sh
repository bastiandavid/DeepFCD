#!/bin/bash

# Simple script for translating the input and output of the generator network to the native t1-space, intensity transformation (zero mean unit variance) and mask generation. Additionally to the difference image, we produce a outlier weight image using mri_robust_register (see ) Also registering MAP morphometric maps, if specified.
# WIP: Should be redone in nipype at some point. Also registration ultimately not necessary if generator trained in native space.
# space in the first place.
# Written by: Bastian David, M.Sc.

# input directories
BASE_DIR=/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/nii

REAL_T1_DIR=${BASE_DIR}/T1
REAL_FLAIR_DIR=${BASE_DIR}/FLAIR
GAN_INPUT_T1_DIR=${BASE_DIR}/gan_input_T1
DIFF_DIR=${BASE_DIR}/diff_real_FLAIR-synth_FLAIR
MAP_DIR=${BASE_DIR}/MAP_DeepFCD

# make output directories
MATRICES_DIR=${BASE_DIR}/matrices
MASK_DIR=${BASE_DIR}/masks
DEEPMEDIC_INPUT=${BASE_DIR}/deepmedic_input
tmp_dir=${BASE_DIR}/tmp
mkdir -p $MATRICES_DIR $MASK_DIR $DEEPMEDIC_INPUT $tmp_dir

# define subjects
#SUBJECTS=$(ls ${T1_DIR}| cut -d'_' -f1)
SUBJECTS=11375

echo ""
echo "#########################################################################"
echo "#################       PRE-DEEPMEDIC PROCESSING       ##################"
echo "#########################################################################"
echo ""
for sbj in $SUBJECTS
do
  echo "Processing $sbj"
  echo "registration"

  flirt -in ${GAN_INPUT_T1_DIR}/${sbj}* -ref ${REAL_T1_DIR}/${sbj}_T1.nii.gz -omat ${MATRICES_DIR}/${sbj}_gan_input_T1_2_T1.mat -nosearch -noresampblur -cost normmi -interp spline

  flirt -in ${DIFF_DIR}/${sbj}* -ref ${REAL_T1_DIR}/${sbj}_T1.nii.gz -applyxfm -init ${MATRICES_DIR}/${sbj}_gan_input_T1_2_T1.mat -nosearch -noresampblur -cost normmi -interp spline -out ${DEEPMEDIC_INPUT}/${sbj}_diff
  echo "fast"
  fast -g -o ${tmp_dir}/${sbj} ${REAL_T1_DIR}/${sbj}_T1.nii.gz

  fslmaths ${tmp_dir}/${sbj}_seg_1 -add ${tmp_dir}/${sbj}_seg_2 ${tmp_dir}/${sbj}_gmwm
  fslmaths ${tmp_dir}/${sbj}_gmwm -fillh ${tmp_dir}/${sbj}_gmwm
  fslmaths ${tmp_dir}/${sbj}_gmwm -kernel sphere 1 -ero ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded
  echo "normalization"
  # normalizing difference
  read -r mean std <<< $(fslstats ${DEEPMEDIC_INPUT}/${sbj}_diff -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${DEEPMEDIC_INPUT}/${sbj}_diff -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded ${DEEPMEDIC_INPUT}/${sbj}_diff

  # normalizing T1
  read -r mean std <<< $(fslstats ${REAL_T1_DIR}/${sbj}* -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${REAL_T1_DIR}/${sbj}* -sub $mean -div $std ${DEEPMEDIC_INPUT}/${sbj}_T1

  # normalizing FLAIR
  read -r mean std <<< $(fslstats ${REAL_FLAIR_DIR}/${sbj}* -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${REAL_FLAIR_DIR}/${sbj}* -sub $mean -div $std ${DEEPMEDIC_INPUT}/${sbj}_FLAIR

  # normalizing junction map
  read -r mean std <<< $(fslstats ${MAP_DIR}/T1_${sbj}_junction_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${MAP_DIR}/T1_${sbj}_junction_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded ${DEEPMEDIC_INPUT}/${sbj}_junction

  # normalizing extension map
  read -r mean std <<< $(fslstats ${MAP_DIR}/T1_${sbj}_extension_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${MAP_DIR}/T1_${sbj}_extension_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded ${DEEPMEDIC_INPUT}/${sbj}_extension

  # normalzing thickness map
  read -r mean std <<< $(fslstats ${MAP_DIR}/T1_${sbj}_thickness_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded -m -s)

  fslmaths ${MAP_DIR}/T1_${sbj}_thickness_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_gmwm_eroded ${DEEPMEDIC_INPUT}/${sbj}_thickness

done
