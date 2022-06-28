# data

## Manually curated dataset related

cleaned_pre_pu_learn_data.csv: the curated dataset used for PU learn
post_pu_learn_data.csv: the updated cleaned_pre_pu_learn_data.csv after validation from PU-Learn
manually_curated_dataset_evaluation_other_synthesis.csv: 55 randomly selected non-solid-state synthesized entries for extraction evaluation
manually_curated_dataset_evaluation_ss_synthesis.csv: 100 randomly selected solid-state synthesized entries for extraction evaluation 

## Outlier detection related

examined_outlier.csv: examined outlier entries from kononova et al's solid-state_dataset_20200713. Dataset link: https://github.com/CederGroupHub/text-mined-synthesis_public

## Feature related

binary_oxide_melting_point.csv: a list of melting point used to create binary oxide melting point features in the paper

## Model related

set_x.json: the files that contain the features and labels for model X
model_x_input.json: the input files of model X at each iterations during hyperparameter tuning
model_x_tuned_input.json: the input files for model X after hyperparameter tuning
non_ss_synthesized_check.csv: a sublist of non-solid-state synthesized entries predicted to be solid-state synthesized by model 1 for model evaluation
hypothetical_check.csv: a sublist of hypothetical entries predicted to be solid-state synthesized by model 1 and 3, which was examined for model evaluation