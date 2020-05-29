#!/bin/bash

# Wrapper for preprocessing_for_deepmedic.sh

# Written by: Bastian David, M.Sc.
echo ""
echo "#########################################################################"
echo "#################       PRE-DEEPMEDIC PROCESSING       ##################"
echo "#########################################################################"
echo ""

# define subjects here
SUBJECTS=$(ls ${T1_DIR}| cut -d'_' -f1)

# temporary directory
BASE_DIR=/home/bdavid/Deep_Learning/data/bonn/FCD/iso_FLAIR/nii
MATRICES_DIR=${BASE_DIR}/matrices
tmp_dir=${BASE_DIR}/tmp

max_cores=$(grep -c ^processor /proc/cpuinfo)
echo "How many cores shall be used? [1-$max_cores]:"
read cores



if [[ "$cores" =~ ^[0-9]+$ ]] && [ "$cores" -ge 1 -a "$cores" -le $max_cores ];
then
    echo ""
    echo "$cores cores will be used."

else
    echo ""
    echo "Invalid input - Terminating script."
    exit 1
fi


echo ""
echo "STARTING PARALLEL PROCESSING"
echo ""
echo "Processing list:"
echo ""


echo $SUBJECTS | xargs -n 1 -P $cores ./preprocessing_for_deepmedic.sh | grep "Processing"

echo ""
echo "All done. Cleaning directory."
rm -rf ${tmp_dir} ${MATRICES_DIR}
