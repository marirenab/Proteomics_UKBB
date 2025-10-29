data <- read.csv(
  "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Protein_abundance_matrix_variance_partitioning_new.csv",
  row.names = 1
)

info <-read.csv("/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Covariates_variancepartitioning.csv")
library("variancePartition")
colnames(data) <- sub("^X", "", colnames(data))
# remove rows with any NA in info
info <- na.omit(info)
rownames(info) <- info$eid

common <- intersect(colnames(data), rownames(info))
data <- data[, common]
info <- info[common, ]



info$Sex <- factor(info$Sex)
info$Ethnicity <- factor(info$Ethnicity)
info$Parkinson_diagnosis <- factor(info$Parkinson_diagnosis)
info$assessment_center <- factor(info$assessment_center)
info$Season <- factor(info$Season)
info$plate <- factor(info$plate)

form <- ~ sample_age + fasting_time + BMI + Age + smoking + alcohol + (1 | Ethnicity) + (1 | Parkinson_diagnosis) + (1 | assessment_center)  + (1 | Season) + (1 | Sex)  + (1 | plate) 

varPart <- fitExtractVarPartModel(data, form, info)
vp <- sortCols(varPart)

# write the full variance partition matrix to CSV
write.csv(vp, file = "/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/variancePartition_results.csv", row.names = TRUE)

pdf("/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/variancePartition_barplot_top10.pdf", width = 10, height = 10)
plotPercentBars(vp[1:10, ])
dev.off()


pdf("/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/variancePartition_summary.pdf", width = 14, height = 6)
plotVarPart(vp)
dev.off()
