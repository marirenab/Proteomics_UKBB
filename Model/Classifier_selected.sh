#!/bin/bash
#PBS -l select=1:ncpus=18:mem=10gb
#PBS -l walltime=8:00:00
#PBS -N modelevaluation
#PBS -e modelevaluationrecall_^array_index^_error
#PBS -o modelevaluationrecall_^array_index^_output

#PBS -J 0-160

module load anaconda3/personal
source ~/anaconda3/etc/profile.d/conda.sh
conda activate test_env


TRAINING_FILES=(
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Training_all_prodromals.csv"

) 


MODEL_TYPES=(
    "RandomForest_balanced"
    "RandomForest_subsample"
    "XGBoost"
    "LightGBM_balanced"
    "BalancedRandomForest"
    "LogisticRegression"
    "LASSORegression"
)

IMPUTERS=(
    "SIMPLEMEDIAN"
)

PROTEIN_FILES=(
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_matched_prodromals_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_residuals_PD_HC_unique_PD_OND_residualised_training_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_matched_prodromals_PD_HC_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_residuals_PD_HC_residualisedhealthycontrol_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_cox_specificPD_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_residuals_PD_HC_unique_residualised_training_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLightGBM_balanced_proteins_matched_baseline_PD_HC_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_unique_residualised_training_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_unique_PD_OND_residualisedhealthycontrol_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_matched_baseline_PD_HC_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_residualisedtraining_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_PD_OND_residualisedtraining_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_PD_OND_residualisedhealthycontrol_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_residualisedhealthycontrol_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_cox_PD_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_cox_specificPD_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_matched_all_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_matched_baseline_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_residuals_PD_HC_unique_PD_OND_residualised_training_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results/PredictionsLASSORegression_proteins_matched_prodromals_PD_HC_unique_SIMPLEMEDIAN_recall_Training_all_prodromals_specificfeatures.txt"
)


METRICS_OPT=("recall")


NUM_TRAINING=${#TRAINING_FILES[@]}
NUM_MODELS=${#MODEL_TYPES[@]}
NUM_IMPUTERS=${#IMPUTERS[@]}
NUM_PROTEINS=${#PROTEIN_FILES[@]}
NUM_METRICS=${#METRICS_OPT[@]}

# Map PBS index -> combo
index=$PBS_ARRAY_INDEX

training_index=$(( index % NUM_TRAINING ))
index=$(( index / NUM_TRAINING ))

model_index=$(( index % NUM_MODELS ))
index=$(( index / NUM_MODELS ))

imputer_index=$(( index % NUM_IMPUTERS ))
index=$(( index / NUM_IMPUTERS ))

protein_index=$(( index % NUM_PROTEINS ))
index=$(( index / NUM_PROTEINS ))

metrics_index=$(( index % NUM_METRICS ))




dataset=${TRAINING_FILES[$training_index]}
model=${MODEL_TYPES[$model_index]}
imputer=${IMPUTERS[$imputer_index]}
protein_file=${PROTEIN_FILES[$protein_index]}
metrics_opt=${METRICS_OPT[$metrics_index]}  


echo "dataset=$dataset"
echo "model=$model"
echo "imputer=$imputer"
echo "protein_file=$protein_file"
echo "metrics_opt=$metrics_opt"

# Check that the files exist
for f in "$dataset" "$protein_file"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: File not found: $f"
        exit 1
    fi
done



python '/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/models.py' "$dataset" "$model" "$protein_file" "$imputer" "$metrics_opt" 

