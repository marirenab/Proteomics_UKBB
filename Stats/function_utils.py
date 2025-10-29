import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
import os
from adjustText import adjust_text
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
path_output= "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Stats/Results"

def filter_patients(disease_df,control_df,visit_df,diagnosis_window):
        disease_df_ = disease_df.copy()
        control_df_ = control_df.copy()

        disease_df_ = pd.merge(disease_df_, visit_df, on="eid")
        disease_df_['time_to_diagnosis'] = pd.to_datetime(disease_df_['date_diagnosis']) - pd.to_datetime(disease_df_['p53_i0'])
        disease_df_['time_to_diagnosis_years'] = disease_df_['time_to_diagnosis'].dt.days / 365.25

        

        if "date_diagnosis" in control_df_.columns:
            control_df_ = pd.merge(control_df_, visit_df, on="eid")
            control_df_['time_to_diagnosis'] = (
                pd.to_datetime(control_df_['date_diagnosis']) - pd.to_datetime(control_df_['p53_i0'])
            )
            control_df_['time_to_diagnosis_years'] = (
                control_df_['time_to_diagnosis'].dt.days / 365.25
            )
        else:
            control_df_['time_to_diagnosis_years'] = np.nan

        if diagnosis_window == "baseline":
            disease_df_ = disease_df_[disease_df_['time_to_diagnosis_years'] <= 2]
            if "date_diagnosis" in control_df_.columns:
                control_df_ = control_df_[control_df_['time_to_diagnosis_years'] <= 2]
            else:
                control_df_ = control_df_
        elif diagnosis_window == "prodromals":
            disease_df_ = disease_df_[disease_df_['time_to_diagnosis_years'] > 2]
            if "date_diagnosis" in control_df_.columns:
                control_df_ = control_df_[control_df_['time_to_diagnosis_years'] > 2]
            else:
                control_df_ = control_df_

        return disease_df_, control_df_




def preprocess_data(disease_df_, control_df, covariates_df,df_expr, diagnosis_window, disease_label, control_label):
        disease_proteins = pd.merge(disease_df_, df_expr, on="eid")[["eid", "protein_abbr", "Expression levels"]]
        control_proteins = pd.merge(control_df, df_expr, on="eid")[["eid", "protein_abbr", "Expression levels"]]
        disease_list = disease_proteins["eid"].unique().tolist()
        control_list = control_proteins["eid"].unique().tolist()
        control_proteins = control_proteins[~control_proteins["eid"].isin(disease_list)]
        covariates_disease = covariates_df[covariates_df["eid"].isin(disease_list)].copy()
        covariates_control = covariates_df[covariates_df["eid"].isin(control_list)].copy()
        covariates_disease["group"] = 1
        covariates_control["group"] = 0
        covariates_all = pd.concat([covariates_disease, covariates_control])
        covariates_all.dropna(inplace=True)
        
        print(f"\n📊 Analysis for {diagnosis_window} between {disease_label} and {control_label}: Number of individuals before matching:")
        print(f"- Disease group: {disease_proteins['eid'].nunique()}")
        print(f"- Control group: {control_proteins['eid'].nunique()}")

        return disease_proteins, control_proteins, covariates_all



def prepare_covariates(disease_eids, control_eids, covariates_all):
        covariates = covariates_all.copy()
        covariates = covariates[covariates["eid"].isin(disease_eids + control_eids)]
        covariates["group"] = covariates["eid"].apply(lambda x: 1 if x in disease_eids else 0)
        covariates.dropna(inplace=True)
        df_encoded = pd.get_dummies(covariates, columns=["Sex", "Ethnicity", "Season"], drop_first=True)
        return df_encoded



def perform_matching(df_encoded, control_proteins, age_tolerance, covariate_cols):
        df = df_encoded.copy()

        # Add protein counts
        protein_count = control_proteins.groupby("eid")["protein_abbr"].nunique().reset_index()
        protein_count.columns = ["eid", "protein_count"]
        df = df.merge(protein_count, on="eid", how="left")
        df["protein_count"] = df["protein_count"].fillna(0)

        # Fit logistic regression for propensity score
        covariate_cols = covariate_cols
        try:
            X = df[covariate_cols].apply(pd.to_numeric, errors="coerce")
            X = sm.add_constant(X)
            y = df["group"]
            model = sm.Logit(y, X).fit(disp=False)
            df["propensity_score"] = model.predict(X)
        except Exception as e:
            print("⚠️ Logistic regression failed. Matching without propensity scores.")
            use_propensity = False
            df["propensity_score"] = 0.5  # neutral constant score for matching

        # Split into treated and control
        treated = df[df["group"] == 1].copy()
        control = df[df["group"] == 0].copy().sort_values(by="protein_count", ascending=False)

        matched_control_rows = []
        matched_treated_rows = []
        used_treated_eids = set()
        used_control_eids = set()

        for _, row in treated.iterrows():
            if row["eid"] in used_treated_eids:
                continue  # Already matched

            available = control[~control["eid"].isin(used_control_eids)]
            available = available[
                (available["Sex_1"] == row["Sex_1"]) &
                (available["Age"].between(row["Age"] - age_tolerance, row["Age"] + age_tolerance))
            ]
            if available.empty:
                continue

            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(available[["propensity_score"]])
            _, nearest_idx = nn.kneighbors([[row["propensity_score"]]])
            matched_row = available.iloc[nearest_idx[0][0]]
            matched_eid = matched_row["eid"]

            if matched_eid in used_control_eids:
                continue

            matched_control_rows.append(matched_row)
            matched_treated_rows.append(row)

            used_control_eids.add(matched_eid)
            used_treated_eids.add(row["eid"])

        # Build matched DataFrames
        matched_controls_df = pd.DataFrame(matched_control_rows)
        matched_treated_df = pd.DataFrame(matched_treated_rows)

        # Create matched pairs
        matched_pairs_df = pd.DataFrame({
            "baseline_eid": matched_treated_df["eid"].values,
            "control_eid": matched_controls_df["eid"].values
        })

        # Combine matched data
        matched_df = pd.concat([matched_treated_df, matched_controls_df], axis=0)

        # === Validation Output ===
        print("\n🔍 Matched Pair Table Summary:")
        print("PD eids:", matched_pairs_df["baseline_eid"].nunique())
        print("Control eids:", matched_pairs_df["control_eid"].nunique())

        print(f"\n✅ Matched {len(matched_pairs_df)} treated-control pairs (total {len(matched_df)} samples).\n")

        print("== Average Age by Group ==")
        print(matched_df.groupby("group")["Age"].mean())

        print("\n== Sex Distribution by Group ==")
        print(matched_df.groupby("group")["Sex_1"].value_counts())

        print("\n== Unique EIDs per Group ==")
        print("PD group:", matched_df[matched_df["group"] == 1]["eid"].nunique())
        print("Control group:", matched_df[matched_df["group"] == 0]["eid"].nunique())

        print("\n== Duplicate usage checks ==")
        print("Duplicated treated eids:", matched_treated_df["eid"].duplicated().sum())
        print("Duplicated control eids:", matched_controls_df["eid"].duplicated().sum())

        return matched_df, matched_pairs_df



def fill_missing_proteins(df, protein_col, eid_col, value_col, group_col):
        df_wide = df.pivot_table(
            index=[eid_col, group_col],
            columns=protein_col,
            values=value_col,
            aggfunc='mean'
        ).reset_index()
        return df_wide



def process_matched_proteins(df_wide, matched_pairs_df):
        def propagate_missing(df_main, df_pairs):
            df_main = df_main.copy()
            df_main.set_index('eid', inplace=True)

            missing_eids = set(df_pairs['baseline_eid']).difference(df_main.index).union(
                set(df_pairs['control_eid']).difference(df_main.index)
            )
            if missing_eids:
                raise KeyError(f"Missing eids in df_main: {missing_eids}")

            for _, row in df_pairs.iterrows():
                eid_1, eid_2 = row['baseline_eid'], row['control_eid']
                for col in df_main.columns.drop('group'):
                    if pd.isna(df_main.at[eid_1, col]):
                        df_main.at[eid_2, col] = np.nan
                    elif pd.isna(df_main.at[eid_2, col]):
                        df_main.at[eid_1, col] = np.nan

            return df_main.reset_index()

        df_wide_cleaned = propagate_missing(df_wide, matched_pairs_df,disease_label,control_label)

        df_long = pd.melt(df_wide_cleaned, id_vars=['eid', 'group'], 
                          var_name='protein', value_name='value')
        df_long["group"] = df_long["group"].replace({1: disease_label, 0: control_label})

        print("== Protein Coverage Summary (after missingness propagation) ==")
        counts = df_long.dropna()
        print("Minimum proteins per eid:", counts['eid'].value_counts().min())
        print("Maximum proteins per eid:", counts['eid'].value_counts().max())
        print("Average proteins per eid:", counts['eid'].value_counts().mean())
        print("Median proteins per eid:", counts['eid'].value_counts().median())
        print("Number of unique proteins measured:", df_long['protein'].nunique())

        return df_long



def perform_stat_tests(df_long, disease_label, control_label):
        results = []
        for protein in df_long["protein"].unique():
            subset = df_long[df_long["protein"] == protein].dropna()

            disease_vals = subset[subset["group"] == disease_label]["value"]
            control_vals = subset[subset["group"] == control_label]["value"]

            n_disease = subset[subset["group"] == disease_label]["eid"].nunique()
            n_control = subset[subset["group"] == control_label]["eid"].nunique()

            if len(disease_vals) == 0 or len(control_vals) == 0:
                continue

            stat, p = mannwhitneyu(disease_vals, control_vals, alternative="two-sided")

            results.append({
                "protein": protein,
                "test_stat": stat,
                "p_value": p,
                "median_diff": disease_vals.median() - control_vals.median(),
                "mean_diff": disease_vals.mean() - control_vals.mean(),
                "n_disease": n_disease,
                "n_control": n_control
            })

        results_df = pd.DataFrame(results)
        results_df["p_value_corrected"] = multipletests(results_df["p_value"], method="bonferroni")[1]
        return results_df


def save_results_and_plot(results_df,path_output,output_prefix):
        results_df["significance"] = "Not Significant"
        results_df.loc[(results_df["p_value_corrected"] < 0.05) & (results_df["mean_diff"] > 0), "significance"] = "Up"
        results_df.loc[(results_df["p_value_corrected"] < 0.05) & (results_df["mean_diff"] < 0), "significance"] = "Down"
        results_df["neg_log10_p"] = -np.log10(results_df["p_value_corrected"])

        plt.figure(figsize=(10, 6))
        colors = {"Up": "red", "Down": "steelblue", "Not Significant": "grey"}
        for label, color in colors.items():
            subset = results_df[results_df["significance"] == label]
            plt.scatter(subset["mean_diff"], subset["neg_log10_p"], label=label, color=color, alpha=0.7, edgecolors="k", s=50)

        # Axes reference lines
        plt.axhline(-np.log10(0.05), linestyle="--", color="black")
        plt.axvline(0, linestyle="--", color="black")

        # Add annotations using adjustText
        texts = []
        top_sig = results_df[
            (results_df["significance"] != "Not Significant") &
            (results_df["neg_log10_p"] > -np.log10(0.05)) 
           
        ]

        for _, row in top_sig.iterrows():
            texts.append(
                plt.text(row["mean_diff"], row["neg_log10_p"], row["protein"], fontsize=7)
            )

        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

        # Labels and save
        plt.xlabel("Mean Difference")
        plt.ylabel("-log10(p-value)")
        plt.legend()
        plt.title("Volcano Plot")
        plt.tight_layout()
        plt.savefig(f"{path_output}/{output_prefix}_volcano.pdf")
        results_df.to_csv(f"{path_output}/{output_prefix}_results.csv", index=False)



def compare_protein_expression(
    disease_df,
    control_df,
    disease_label,
    control_label,
    df_expr,
    visit_df,
    covariates_df,
    covariate_cols= None,
    output_prefix="comparison",
    diagnosis_window="all",  # options: "baseline", "prodromals", "all"
    age_tolerance=1,
    path_output =path_output):


    disease_df_, control_df = filter_patients()
    disease_proteins, control_proteins, covariates_all = preprocess_data(disease_df_, control_df, covariates_df)
    covariates_encoded = prepare_covariates(disease_df_["eid"].tolist(), control_df["eid"].tolist(), covariates_all)
    matched_df, matched_pairs_df = perform_matching(covariates_encoded, df_expr, age_tolerance, covariate_cols)

    # Combine proteins and fill missing
    all_proteins = pd.concat([disease_proteins, control_proteins])
    all_proteins["group"] = all_proteins["eid"].map(dict(zip(matched_df["eid"], matched_df["group"])))
    df_wide = fill_missing_proteins(all_proteins, "protein_abbr", "eid", "Expression levels", "group")

    # Process missingness and convert to long
    df_long = process_matched_proteins(df_wide, matched_pairs_df)

    # Run stats and plot
    results_df = perform_stat_tests(df_long,disease_label, control_label)
    save_results_and_plot(results_df)

    return results_df