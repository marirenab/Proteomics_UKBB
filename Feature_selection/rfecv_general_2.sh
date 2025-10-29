#!/bin/bash
#PBS -l select=1:ncpus=40:mem=10gb
#PBS -l walltime=8:00:00
#PBS -N rfecv_neuro
#PBS -e rfecv_healthy_lightgbm^array_index^_stderr
#PBS -o fecv_healthy_lightgbm_^array_index^_stout
#PBS -J 0-2
# ---------------------------
# Load Anaconda and activate environment
# ---------------------------
module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env

# ---------------------------
# Define paired datasets and feature lists
# ---------------------------
DATASETS=(
  
    "Training_residualstraining_neurodegenerative.csv"
    "Training_residualshealthycontrol_neurodegenerative.csv"
    "Training_residualsealthycontrol_healthycontrol.csv"
)

FEATURES_FILES=(

"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/Results/Specificfeatures_LASSORegression_Training_residualstraining_neurodegenerative.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/Results/Specificfeatures_LASSORegression_Training_residualshealthycontrol_neurodegenerative.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/Results/Specificfeatures_LASSORegression_Training_residualshealthycontrol_healthycontrol.txt"
)



MODEL="LASSORegression"
i=${PBS_ARRAY_INDEX:-0}
dataset="${DATASETS[$i]}"
features_file="${FEATURES_FILES[$i]}"

echo "Running model: $MODEL"
echo "Dataset: $dataset"
echo "Features: $features_file"
echo "Job index: $PBS_ARRAY_INDEX"

# ---------------------------
# Run Python script
# ---------------------------
python "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/feature_rfecv_general.py" \
    --input "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/$dataset" \
    --input_features "$features_file" \
    --input_clf "$MODEL"
