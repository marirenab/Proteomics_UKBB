#!/bin/bash
#PBS -l select=1:ncpus=50:mem=100gb
#PBS -l walltime=8:00:00
#PBS -N rfecv_healthy
#PBS -e FS_^array_index^_stderr
#PBS -o FS^array_index^_stout
#PBS -J 0-7

# Load Anaconda and activate environment
module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env

BASE_PATH="/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/"

# --- Datasets with full path ---
DATASETS=(
    "${BASE_PATH}Training_residualshealthycontrol_healthycontrol.csv"
    "${BASE_PATH}Training_residualstraining_healthycontrol.csv"
    "${BASE_PATH}Training_residualstraining_neurodegenerative.csv"
    "${BASE_PATH}Training_residualshealthycontrol_neurodegenerative.csv"
)

MODELS=("LightGBM_balanced" "LASSORegression")

# --- Map PBS_ARRAY_INDEX to combination ---
index=$PBS_ARRAY_INDEX

dataset_index=$(( index % ${#DATASETS[@]} ))
index=$(( index / ${#DATASETS[@]} ))

model_index=$(( index % ${#MODELS[@]} ))

dataset_name=${DATASETS[$dataset_index]}
model_name=${MODELS[$model_index]}

echo "Dataset: $dataset_name"
echo "Model: $model_name"

# --- Run Python script ---
python '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/FS.py' "$dataset_name" "$model_name"
