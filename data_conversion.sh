#!/bin/bash

# Set the base directory where the folders are located
BASE_DIR="/home/rklab1/Documents/Raw_data_MRS"  # Replace with your actual directory path

# Loop through each directory within the BASE_DIR
for folder in "$BASE_DIR"/*/; do
  # Check if it's a directory
  if [ -d "$folder" ]; then
    # Get the folder name without the path
    folder_name=$(basename "$folder")
    
    # Run the spec2nii command with the current folder
    spec2nii bruker -m FID "$folder"
    
    # Print a message for each processed folder (optional)
    echo "Processed folder: $folder_name"
  fi
done

