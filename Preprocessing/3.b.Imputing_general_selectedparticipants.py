# %%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import joblib
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=10)

path = "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/"

df = pd.read_table("/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/omics/olink_data.txt")
df = df[df["ins_index"]==0]
codings = pd.read_csv('/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/codings/coding143.tsv', sep='\t')
codings[['protein_abbr','protein_name']] = codings['meaning'].str.split(';',expand=True).rename(columns={0:'protein_abbr',1:'protein_name'})
codings.rename(columns={"coding":"protein_id"},inplace=True)
df_merged = pd.merge(codings, df, on=["protein_id"])
df_merged.rename(columns={"result":"Expression levels"},inplace=True)
df_merged = df_merged[["eid","protein_abbr","Expression levels"]]



df_consortium = pd.read_csv("/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/data_2025/proteomics/olink_consortium.csv")
df_consortium = df_consortium.dropna(thresh=2)
consortium_picked_participants = df_consortium["eid"].tolist()


visit_dates= pd.read_csv("/rds/general/user/meb22/projects/ukbiobank/live/ukbiobank/data_2025/Visit_dates.csv").iloc[:,1:]
covid_imaging_visit_participants = visit_dates.dropna(subset="p53_i3")["eid"].tolist()
nongeneralpopulation_participants = covid_imaging_visit_participants +consortium_picked_participants



withdrawn = pd.read_csv(f"{path}withdrawnparticipants.csv")
first_col_name = withdrawn.columns[0]
ids_withdrawn_list = [first_col_name] + withdrawn.iloc[:, 0].tolist()
df_merged = df_merged[~df_merged["eid"].isin(ids_withdrawn_list)]
df_merged = df_merged.pivot_table(index=['eid'], columns='protein_abbr', values='Expression levels').reset_index()

df_rest = df_merged[df_merged["eid"].isin(nongeneralpopulation_participants)]
df_rest = df_rest.set_index(["eid"])

df_merged = df_merged[~df_merged["eid"].isin(nongeneralpopulation_participants)]
df_merged = df_merged.set_index(["eid"])

protein_cols = np.loadtxt(f"{path}columns_30pd.txt", dtype=str).tolist()
len(protein_cols)



protein_data_imputed_general = pd.DataFrame(imputer.fit_transform(df_merged[protein_cols]), columns=protein_cols, index=df_merged.index)
protein_data_imputed_general.to_csv(f"{path}General_population_imputed_lesscols.csv")

protein_data_imputed_rest= pd.DataFrame(imputer.transform(df_rest[protein_cols]), columns=protein_cols, index=df_rest.index)
protein_data_imputed_rest.to_csv(f"{path}Consortium_covid_population_imputed_lesscols.csv")


