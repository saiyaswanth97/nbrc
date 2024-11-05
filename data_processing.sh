#!/bin/bash

# Set the base directory where the folders are located
BASE_DIR="/home/rklab1/Documents/Raw_data_MRS"
BASE_DIR="/home/smummaneni/nbrc/data/new/Raw_data_MRS"
OUT_DIR=$BASE_DIR/processed_data
mkdir -p $OUT_DIR

# Loop through to get the .nii.gz files
for i in $(ls $BASE_DIR); do
  # Loop through the files in the directory and check if the file is Nifti file
  if [[ $i == *.nii.gz ]]; then
    # Get the file name without the extension
    # FID_118W_40.nii.gz -> 118W_40 (FILE_NAME)
    FILE_NAME=$(echo $i | cut -d'.' -f1)
    FILE_NAME=$(echo $FILE_NAME | cut -d'_' -f2,3)
    INPUT_FILE=$BASE_DIR/$i

    # Create a folder with the file name in the output directory
    SAMPLE_OUT_FOLDER=$OUT_DIR/$FILE_NAME
    mkdir -p $SAMPLE_OUT_FOLDER

    # For individual processing using fsl_mrs_proc
    fsl_mrs_proc remove --file $INPUT_FILE --out $SAMPLE_OUT_FOLDER --filename removed
    fsl_mrs_proc apodize --filter exp --amount 25 --file $INPUT_FILE --out $SAMPLE_OUT_FOLDER --filename smoothed
    fsl_mrs_proc fixed_phase --p0 90 --file $INPUT_FILE --out $SAMPLE_OUT_FOLDER --filename phase_corrected 
    fsl_mrs_proc model --ppm 1.5 4.0 --file $INPUT_FILE --out $SAMPLE_OUT_FOLDER --filename peak

    # For combined processing using fsl_mrs_proc
    # Fixed phase (90), apodize (exp, 25), remove
    file_processed=$SAMPLE_OUT_FOLDER/fully_processed.nii.gz
    fsl_mrs_proc fixed_phase --p0 90 --file $INPUT_FILE --out $SAMPLE_OUT_FOLDER --filename fully_processed
    fsl_mrs_proc apodize --filter exp --amount 25 --file $file_processed --out $SAMPLE_OUT_FOLDER --filename fully_processed
    fsl_mrs_proc remove --file $file_processed --out $SAMPLE_OUT_FOLDER --filename fully_processed
    fsl_mrs_proc model --file $file_processed --out $SAMPLE_OUT_FOLDER --filename peak_filtered --ppm 1.5 4.0

    # Run fitting
    fsl_mrs --data $file_processed --out $SAMPLE_OUT_FOLDER/out --basis /home/smummaneni/nbrc/data/new/9.4t/gamma_press_te15_9.4t_v1.basis 
  fi
done
