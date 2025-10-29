# 🧬 UK Biobank OLINK Proteomics Analysis

This repository contains the code for **preprocessing**, **feature selection**, and **machine learning modelling** of the **UK Biobank OLINK proteomics** dataset.  
The analysis aims to identify predictive protein biomarkers related to Parkinson’s Disease (PD) and other neurodegenerative conditions.

---

## 📂 Repository Structure

### **`preprocessing/`**
Contains scripts for:
- Extraction and cleaning of **clinical and proteomic data**  
- Creation of **training and test sets**  
- **Variance partitioning analysis** of OLINK data
- Preprocessing of OLINK data
- Imputation of OLINK data
- Correction for **confounders**

### **`Stats/`**
Contains statistical analysis scripts, for group comparisons between **PD**, **healthy controls (HC)**, and other **neurodegenerative disorders****. The methods include:
- **Matched analyses** controlling for age, sex, and confounders
- **Mann–Whitney U tests** on residualised data 
- **Cox regression** for survival analysis  



### **`Feature_selection/`**
Includes machine learning–based feature selection methods:
- **LightGBM** and **Lasso Regression** wrapper method. 
- **Boruta** using both models. 
- **Recursive Feature Elimination with Cross-Validation (RFECV)** using both models. 


### **`Model/`**
Contains scripts for:
- **Automated job submission** of different model configurations
- Model training on **different feature selection sets**
- Model training on either prodromals alone or on prodromals + baseline PD.
- Evaluation for incident PD prediction using various **optimisation metrics** 


### **`classification/`**
Dedicated to classification analyses, including:
- Data extraction and preprocessing for **diagnosed PD cases**  
- Model training and evaluation for **disease classification** tasks

---

## ⚙️ Dependencies

This repository uses **Python (3.11)** and relies on the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `lightgbm`
- `xgboost`
- `matplotlib`
- `seaborn`
- `eboruta`
- `statsmodels`


---
