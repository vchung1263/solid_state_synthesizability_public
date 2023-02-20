# Data

## Manually curated dataset related

- manually_curated_dataset_pre_pulearn.csv: the curated dataset used for PU learn
- manually_curated_dataset_evaluation_other_synthesis.csv: 55 randomly selected non-solid-state synthesized entries for manual extraction evaluation
- manually_curated_dataset_evaluation_ss_synthesis.csv: 100 randomly selected solid-state synthesized entries for manual extraction evaluation 

## Outlier detection related

- Kononova_ss_ternary_oxide_examined_outlier.csv: The examined ternary oxide outlier entries from the kononova et al's solid-state_dataset_20200713. Dataset link: https://github.com/CederGroupHub/text-mined-synthesis_public

## Feature related

- binary_oxide_melting_point.csv: a list of melting point used to create the binary oxide melting point features in the paper

## Model related

- set_x.json: the files that contain the features and labels for model X
- model_x_input.json: the input files of model X at each iterations during hyperparameter tuning
- model_x_tuned_input.json: the input files for model X after hyperparameter tuning
- pu_learn_non_ss_synthesized_check.csv: a sublist of non-solid-state synthesized entries predicted to be solid-state synthesized by model 1 for model evaluation
- pu_learn_hypothetical_check.csv: a sublist of hypothetical entries predicted to be solid-state synthesized by model 1 and 3, which was examined for model evaluation