#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 
# N. Wang, 27/07/2024

echo 'Creating UD_RM environment'

# Ensure conda is initialized
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    echo 'Initializing conda...'
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo 'Conda initialization script not found. Please check the path.'
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo 'Conda could not be found. Please ensure it is installed and the path is correct.'
    exit 1
fi

# Create conda environment
echo 'Creating conda environment from environment.yml...'
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UD_RM
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'

