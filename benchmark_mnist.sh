#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --time=96:00:00
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition=nms_research

python mnist_test.py --where=distant --dataset=mnist_dvs_10 --num_ite=5