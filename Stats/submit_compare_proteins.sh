#!/bin/bash
#PBS -l select=1:ncpus=50:mem=50gb
#PBS -l walltime=8:00:00
#PBS -N compare_proteins
#PBS -e compare_proteins_^array_index^_error
#PBS -o compare_proteins_^array_index^^_output
#PBS -J 0-8

module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env

# Define comparisons: disease vs control
DISEASES=("PD" "PD" "OND")
CONTROLS=("OND" "HC" "HC")

# Diagnosis windows
WINDOWS=("baseline" "prodromals" "all")

NUM_COMPARISONS=${#DISEASES[@]}
NUM_WINDOWS=${#WINDOWS[@]}

# Map PBS_ARRAY_INDEX -> comparison + window
index=$PBS_ARRAY_INDEX
comparison_index=$(( index % NUM_COMPARISONS ))
window_index=$(( index / NUM_COMPARISONS ))

disease_label=${DISEASES[$comparison_index]}
control_label=${CONTROLS[$comparison_index]}
diagnosis_window=${WINDOWS[$window_index]}

echo "Running comparison: $disease_label vs $control_label | window: $diagnosis_window"


python /rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Stats/stats_all.py  "$control_label" "$disease_label" "$diagnosis_window"
