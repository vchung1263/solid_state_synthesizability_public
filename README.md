# Solid State Synthesizability Prediction
This repository contains the code, data and result of our manuscript.

## Developers
Vincent Chung

## Dependencies
- Python3
- Numpy
- Pandas
- Scikit-learn
- Scikit-optimize
- Lightgbm
- Shap

## Installation
```
git clone https://github.com/vchung1263/solid_state_synthesizability_public.git
```
Note that the code have only been tested on running in Windows.

## Description

The pu_learn_model.py contains the PU-Learning models used in the manuscript. It is a modified code of Jang et al.. Link to the original code repository: https://github.com/kaist-amsg/Synthesizability-PU-CGCNN. Model_x.py and baseline_model.py are the scripts used to predict the synthesizability score of the compositions using the input files in ./folder.

- The ./data folder contains the input files used by the models and the csv files for the results in the paper (more info in ./data)
- The ./result folder contains 2 types of files: the csv files contain the tuned model input and predicted synthesizability score, while the pkl files contain the output of of the models
- The ./checkpoint folder contains the checkpoint of the hyperparameter tuning for each models, which contains the hyperparameters and model performance of each iteration.

## Publication
Vincent Chung, Aron Walsh, and David Payne. Solid-State Synthesizability Predictions Using Positive-Unlabelled Learning from Curated Literature Data, In preparation

