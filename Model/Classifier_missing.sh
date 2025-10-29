#!/bin/bash
#PBS -l select=1:ncpus=20:mem=10gb
#PBS -l walltime=8:00:00
#PBS -N modelevaluation_missing
#PBS -e modelevaluation_missing_^array_index^_error
#PBS -o modelevaluation_missing_^array_index^_output
#PBS -J 0-83

module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env

TRAINING_FILE="/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Training_all_prodromals.csv"
IMPUTER="SIMPLEMEDIAN"
METRICS_OPT="recall"

# Read the specific line for this array index
COMBO=$(sed -n "$((PBS_ARRAY_INDEX+1))p" /rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/missing_combinations.txt)
MODEL=$(echo $COMBO | cut -d' ' -f1)
PROTEIN_FILE=$(echo $COMBO | cut -d' ' -f2)

echo "Running model=$MODEL with protein_file=$PROTEIN_FILE"


# Run your Python script
python '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/models.py' \
    "$TRAINING_FILE" "$MODEL" "$PROTEIN_FILE" "$IMPUTER" "$METRICS_OPT"
