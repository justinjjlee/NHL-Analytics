#!/bin/bash

# Activate your conda environment
conda activate myenv

# Generate Conda Requirements
conda list --export | awk -F= '/^[^#]/ && !/pypi/ {print $1"=="$2}' > conda_requirements.txt

# Generate Pip Requirements
pip list --format=freeze > pip_requirements.txt

echo "Separated conda and pip packages into conda_requirements.txt and pip_requirements.txt"
