#!/bin/bash
#PBS -l select=1:ncpus=90:mem=90gb
#PBS -l walltime=8:00:00
#PBS -N imputationgeneral
#PBS -e imputationgeneral.e
#PBS -o imputationgeneral.o


# Load Anaconda and activate the environment
module load anaconda3/personal
module ~/anaconda3/etc/profile.d/conda.sh
source activate test_env


# Run the Python script with the model name
python '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/3.b.Imputing_general_selectedparticipants.py'
