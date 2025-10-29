#!/bin/bash
#PBS -l select=1:ncpus=20:mem=10gb
#PBS -l walltime=8:00:00
#PBS -N modelevaluationbaseline
#PBS -e modelevaluationbaseline_^array_index^_error
#PBS -o modelevaluationbaselinee_^array_index^_output
#PBS -J 0-47

module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env


TRAINING_FILES_PRODROMALS=(
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Training_all_prodromals.csv"
) 


TRAINING_FILES_BASELINES=(
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Training_baselinePD.csv"

) 



MODEL_TYPES=(
    "RandomForest_balanced"
    "RandomForest_subsample"
    "LightGBM_balanced"
    "BalancedRandomForest"
    "LogisticRegression"
    "LASSORegression"
)

IMPUTERS=(
    "SIMPLEMEDIAN"
)

PROTEIN_FILES=(
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_all_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_cox_specificPD_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_baseline_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_baseline_PD_HC_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_all_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_cox_specificPD_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_baseline_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaseline/PredictionsLightGBM_balanced_proteins_matched_baseline_PD_HC_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
)

METRICS_OPT=("recall")


# lengths
NUM_TRAINING=${#TRAINING_FILES_PRODROMALS[@]}   # pairs!
NUM_MODELS=${#MODEL_TYPES[@]}
NUM_IMPUTERS=${#IMPUTERS[@]}
NUM_PROTEINS=${#PROTEIN_FILES[@]}
NUM_METRICS=${#METRICS_OPT[@]}

# map PBS index -> combo
index=$PBS_ARRAY_INDEX
echo "PBS_ARRAY_INDEX=$index"

training_index=$(( index % NUM_TRAINING ))
index=$(( index / NUM_TRAINING ))

model_index=$(( index % NUM_MODELS ))
index=$(( index / NUM_MODELS ))

imputer_index=$(( index % NUM_IMPUTERS ))
index=$(( index / NUM_IMPUTERS ))

protein_index=$(( index % NUM_PROTEINS ))
index=$(( index / NUM_PROTEINS ))

metrics_index=$(( index % NUM_METRICS ))

dataset_prodromal=${TRAINING_FILES_PRODROMALS[$training_index]}
dataset_baseline=${TRAINING_FILES_BASELINES[$training_index]}
model=${MODEL_TYPES[$model_index]}
imputer=${IMPUTERS[$imputer_index]}
protein_file=${PROTEIN_FILES[$protein_index]}
metrics_opt=${METRICS_OPT[$metrics_index]}

echo "dataset_prodromal=$dataset_prodromal"
echo "dataset_baseline=$dataset_baseline"
echo "model=$model"
echo "imputer=$imputer"
echo "protein_file=$protein_file"
echo "metrics_opt=$metrics_opt"

# check that files exist
for f in "$dataset_prodromal" "$dataset_baseline" "$protein_file"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done

# run Python with the pair
python '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/models_addbaseline.py' \
    "$dataset_prodromal" "$dataset_baseline" "$model" "$protein_file" "$imputer" "$metrics_opt"

