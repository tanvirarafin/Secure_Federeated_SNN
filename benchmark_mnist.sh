#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=nms_research

python mnist_test.py
