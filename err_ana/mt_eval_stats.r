library(lme4)
library(dplyr)

facts = read.table("stats_data.tab", header=TRUE, sep="\t", quote="")

# Split data into three datasets based on model_name
for (model in unique(facts$model_name)) {
  assign(paste0("data_", model), filter(facts, model_name == model))
}

# Compute mixed effects regression for a model with random effect for s_id,
# fixed effects for lexicon, dependency, constructions, conllu, predicting BERTScore_F1
make_lmer <- function(data) {
  model <- lmer(BERTScore_F1 ~ lexicon + syntax + (1 | sid), data = data)
  print(summary(model))
  drop1(model, test="Chisq")
}


for (model in unique(facts$model_name)) {
  data <- get(paste0("data_", model))
  drop1_res <- make_lmer(data)
  print(paste("Model:", model))
  print(drop1_res)
}
