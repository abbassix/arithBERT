#!/bin/bash

# Define the parent directory
parent_dir="../"

# List of directories to check and create if they do not exist
directories=("datasets" "models" "results")

# Loop through each directory in the list
for dir in "${directories[@]}"; do
  # Check if the directory does not exist
  if [ ! -d "${parent_dir}${dir}" ]; then
    echo "Directory ${dir} does not exist. Creating..."
    mkdir "${parent_dir}${dir}"
  else
    echo "Directory ${dir} exists."
  fi
done
