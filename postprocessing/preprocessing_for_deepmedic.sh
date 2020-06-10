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
GAN_TARGET_FLAIR_DIR=${BASE_DIR}/gan_target_FLAIR
SYNTH_FLAIR_DIR=${BASE_DIR}/synth_FLAIR
DIFF_DIR=${BASE_DIR}/diff_real_FLAIR-synth_FLAIR
MAP_DIR=${BASE_DIR}/MAP_DeepFCD
ROI_DIR=${BASE_DIR}/ROI

# make output directories
MATRICES_DIR=${BASE_DIR}/matrices
DEEPMEDIC_INPUT=${BASE_DIR}/deepmedic_input
tmp_dir=${BASE_DIR}/tmp
mkdir -p $MATRICES_DIR $DEEPMEDIC_INPUT $tmp_dir

# define subjects (not necessary if using parallelized wrapper script)
#SUBJECTS=$(ls ${T1_DIR}| cut -d'_' -f1)
SUBJECTS=$1

# define ROIs not being purged (filtering out mostly subcortical structures in this step)
rois="3 2 24 41 42 77 78 79 80 81 82 100 109"

for sbj in $SUBJECTS
do
  echo "Processing $sbj"

  flirt -in ${REAL_T1_DIR}/${sbj}_T1.nii.gz -ref ${REAL_T1_DIR}/${sbj}_T1.nii.gz -applyisoxfm 0.8 -nosearch -noresampblur -cost normmi -interp spline -out ${tmp_dir}/${sbj}_T1

  flirt -in ${REAL_FLAIR_DIR}/${sbj}_FLAIR.nii.gz -ref ${tmp_dir}/${sbj}_T1 -omat ${MATRICES_DIR}/${sbj}_FLAIR_2_T1.mat -out ${tmp_dir}/${sbj}_FLAIR -noresampblur -interp spline

  flirt -in ${ROI_DIR}/${sbj}_roi -ref ${tmp_dir}/${sbj}_T1 -applyxfm -init ${MATRICES_DIR}/${sbj}_FLAIR_2_T1.mat -out ${DEEPMEDIC_INPUT}/${sbj}_roi -interp nearestneighbour

  flirt -in ${GAN_INPUT_T1_DIR}/${sbj}_* -ref ${tmp_dir}/${sbj}_T1 -omat ${MATRICES_DIR}/${sbj}_gan_input_T1_2_T1.mat -nosearch -noresampblur -cost normmi -interp spline

  flirt -in ${DIFF_DIR}/${sbj}_* -ref ${tmp_dir}/${sbj}_T1 -applyxfm -init ${MATRICES_DIR}/${sbj}_gan_input_T1_2_T1.mat -nosearch -noresampblur -cost normmi -interp spline -out ${DEEPMEDIC_INPUT}/${sbj}_diff

  bet ${tmp_dir}/${sbj}_T1 ${tmp_dir}/${sbj}_bet_T1 -R
  fast -g -o ${tmp_dir}/${sbj} ${tmp_dir}/${sbj}_bet_T1

  fslmaths ${tmp_dir}/${sbj}_seg_1 -add ${tmp_dir}/${sbj}_seg_2 ${tmp_dir}/${sbj}_gmwm
  fslmaths ${tmp_dir}/${sbj}_gmwm -fillh ${tmp_dir}/${sbj}_gmwm
  fslmaths ${tmp_dir}/${sbj}_gmwm -kernel sphere 1 -ero ${tmp_dir}/${sbj}_gmwm_eroded

  samseg --t1w ${tmp_dir}/${sbj}_T1.nii.gz --flair ${tmp_dir}/${sbj}_FLAIR.nii.gz --refmode t1w --o ${tmp_dir}/${sbj} --no-save-warp --threads 1 --pallidum-separate

  mri_label2vol --seg ${tmp_dir}/${sbj}/seg.mgz --temp ${tmp_dir}/${sbj}_T1.nii.gz --o ${tmp_dir}/${sbj}/seg_reg.nii --regheader ${tmp_dir}/${sbj}/seg.mgz

  fslmaths ${tmp_dir}/${sbj}/seg_reg.nii -mul 0 ${tmp_dir}/${sbj}_only_cortical_structures

  for roi in $rois
  do

    fslmaths ${tmp_dir}/${sbj}/seg_reg.nii -thr ${roi} -uthr ${roi} -bin ${tmp_dir}/${sbj}_roi_tmp
    fslmaths ${tmp_dir}/${sbj}_only_cortical_structures -add ${tmp_dir}/${sbj}_roi_tmp ${tmp_dir}/${sbj}_only_cortical_structures

  done

  fslmaths ${tmp_dir}/${sbj}_gmwm_eroded -mul ${tmp_dir}/${sbj}_only_cortical_structures ${tmp_dir}/${sbj}_gmwm_eroded

  fslmaths ${tmp_dir}/${sbj}_gmwm_eroded -kernel sphere 1 -ero ${tmp_dir}/${sbj}_gmwm_eroded_ero
  fslmaths ${tmp_dir}/${sbj}_gmwm_eroded_ero -kernel sphere 1 -dilF ${DEEPMEDIC_INPUT}/${sbj}_mask

  # intermediate cleaning
  #rm -rf ${tmp_dir}/${sbj}

  # normalizing difference
  read -r mean std <<< $(fslstats ${DEEPMEDIC_INPUT}/${sbj}_diff -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${DEEPMEDIC_INPUT}/${sbj}_diff -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_mask ${DEEPMEDIC_INPUT}/${sbj}_diff

  # normalizing T1
  read -r mean std <<< $(fslstats ${tmp_dir}/${sbj}_T1 -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/${sbj}_T1 -sub $mean -div $std ${DEEPMEDIC_INPUT}/${sbj}_T1

  # normalizing FLAIR
  read -r mean std <<< $(fslstats ${tmp_dir}/${sbj}_FLAIR -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/${sbj}_FLAIR -sub $mean -div $std ${DEEPMEDIC_INPUT}/${sbj}_FLAIR

  # normalizing junction map
  imcp ${MAP_DIR}/T1_${sbj}_junction_z_score ${tmp_dir}/T1_${sbj}_junction_z_score
  fslcpgeom ${REAL_T1_DIR}/${sbj}_T1 ${tmp_dir}/T1_${sbj}_junction_z_score

  flirt -in ${tmp_dir}/T1_${sbj}_junction_z_score -ref ${tmp_dir}/T1_${sbj}_junction_z_score -applyisoxfm 0.8 -nosearch -noresampblur -cost normmi -interp spline -out ${tmp_dir}/T1_${sbj}_junction_z_score

  read -r mean std <<< $(fslstats ${tmp_dir}/T1_${sbj}_junction_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/T1_${sbj}_junction_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_mask ${DEEPMEDIC_INPUT}/${sbj}_junction

  # normalizing extension map
  imcp ${MAP_DIR}/T1_${sbj}_extension_z_score ${tmp_dir}/T1_${sbj}_extension_z_score
  fslcpgeom ${REAL_T1_DIR}/${sbj}_T1 ${tmp_dir}/T1_${sbj}_extension_z_score

  flirt -in ${tmp_dir}/T1_${sbj}_extension_z_score -ref ${tmp_dir}/T1_${sbj}_extension_z_score -applyisoxfm 0.8 -nosearch -noresampblur -cost normmi -interp spline -out ${tmp_dir}/T1_${sbj}_extension_z_score

  read -r mean std <<< $(fslstats ${tmp_dir}/T1_${sbj}_extension_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/T1_${sbj}_extension_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_mask ${DEEPMEDIC_INPUT}/${sbj}_extension

  # normalzing thickness map
  imcp ${MAP_DIR}/T1_${sbj}_thickness_z_score ${tmp_dir}/T1_${sbj}_thickness_z_score
  fslcpgeom ${REAL_T1_DIR}/${sbj}_T1 ${tmp_dir}/T1_${sbj}_thickness_z_score

  flirt -in ${tmp_dir}/T1_${sbj}_thickness_z_score -ref ${tmp_dir}/T1_${sbj}_thickness_z_score -applyisoxfm 0.8 -nosearch -noresampblur -cost normmi -interp spline -out ${tmp_dir}/T1_${sbj}_thickness_z_score

  read -r mean std <<< $(fslstats ${tmp_dir}/T1_${sbj}_thickness_z_score -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/T1_${sbj}_thickness_z_score -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_mask ${DEEPMEDIC_INPUT}/${sbj}_thickness

  # creating weight map using mri_robust_register
  mri_robust_register --mov ${GAN_TARGET_FLAIR_DIR}/${sbj}_* --dst ${SYNTH_FLAIR_DIR}/${sbj}_* --lta ${tmp_dir}/${sbj}.lta --weights ${tmp_dir}/${sbj}_weights.nii --satit

  flirt -in ${tmp_dir}/${sbj}_weights.nii -ref ${tmp_dir}/${sbj}_T1 -applyxfm -init ${MATRICES_DIR}/${sbj}_gan_input_T1_2_T1.mat -nosearch -noresampblur -cost normmi -interp spline -out ${tmp_dir}/${sbj}_weights_reg

  read -r mean std <<< $(fslstats ${tmp_dir}/${sbj}_weights_reg -k ${DEEPMEDIC_INPUT}/${sbj}_mask -m -s)

  fslmaths ${tmp_dir}/${sbj}_weights_reg -sub $mean -div $std -mul ${DEEPMEDIC_INPUT}/${sbj}_mask ${DEEPMEDIC_INPUT}/${sbj}_weights

  # cleaning up temporary directory
  rm -rf ${tmp_dir}/${sbj}_* ${tmp_dir}/${sbj}

done
