# %%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.impute import KNNImputer

# %%
path = "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/"

# %%
df = pd.read_table("/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/omics/olink_data.txt")
df = df[df["ins_index"]==0]
codings = pd.read_csv('/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/codings/coding143.tsv', sep='\t')
codings[['protein_abbr','protein_name']] = codings['meaning'].str.split(';',expand=True).rename(columns={0:'protein_abbr',1:'protein_name'})
codings.rename(columns={"coding":"protein_id"},inplace=True)
df_merged = pd.merge(codings, df, on=["protein_id"])
df_merged.rename(columns={"result":"Expression levels"},inplace=True)
df_merged = df_merged[df_merged["ins_index"] == 0]
df_merged = df_merged[["eid","protein_abbr","Expression levels"]]

# %%
withdrawn = pd.read_csv(f"{path}withdrawnparticipants.csv")
first_col_name = withdrawn.columns[0]
ids_withdrawn_list = [first_col_name] + withdrawn.iloc[:, 0].tolist()
df_merged = df_merged[~df_merged["eid"].isin(ids_withdrawn_list)]

# %%`
df_pivot = df_merged.pivot_table(index=['eid'], columns='protein_abbr', values='Expression levels').reset_index()

# %%
training_set = pd.read_csv(f"{path}Training_all.csv")
testing_set = pd.read_csv(f"{path}Testing_all.csv")
comorbid = pd.read_csv(f"{path}PD_comorbidONDProteomics.csv")

#Read columns that should be imputed (the rest are missing in more than 30% of PD participants)
protein_cols = np.loadtxt(f"{path}columns_30pd.txt", dtype=str).tolist()
len(protein_cols)

# %%
training_set = training_set.set_index(["eid"])
testing_set= testing_set.set_index(["eid"])
comorbid = comorbid.set_index(["eid"])

print("Training set:", training_set.shape)

# %%
training_set = training_set[protein_cols]
testing_set= testing_set[protein_cols]
comorbid = comorbid[protein_cols]


# %%
imputer = KNNImputer(n_neighbors=10)

protein_data_imputed_training = pd.DataFrame(imputer.fit_transform(training_set[protein_cols]), columns=protein_cols, index=training_set.index)
protein_data_imputed_training.to_csv(f"{path}Training_data_imputed_lesscols.csv")
print("Training set after imputation:", protein_data_imputed_training.shape)

protein_data_imputed_testing = pd.DataFrame(imputer.transform(testing_set[protein_cols]), columns=protein_cols, index=testing_set.index)
protein_data_imputed_testing.to_csv(f"{path}Testing_data_imputed_lesscols.csv")

protein_data_imputed_comorb = pd.DataFrame(imputer.transform(comorbid[protein_cols]), columns=protein_cols, index=comorbid.index)
protein_data_imputed_comorb.to_csv(f"{path}Comorbidond_data_imputed_lesscols.csv")


# %%



