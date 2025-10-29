#!/bin/bash
#PBS -l select=1:ncpus=15:mem=200gb
#PBS -l walltime=8:00:00
#PBS -N variance
#PBS -e variance_stderr
#PBS -o variance_stout


# Load Anaconda and activate the environment
source /rds/general/user/meb22/home/miniforge3/bin/activate renv




# Run the Python script with the model name
Rscript '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Variance_partitioning.R' 

