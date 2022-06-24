# General
import numpy as np
import pandas as pd
import pickle
from pu_learn_model import PULearner
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, recall_score, roc_curve
from sklearn.feature_selection import SelectKBest

# skopt
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, DeltaYStopper
from skopt import load

random_state = 42

search_space = [Integer(1, 400, name='max_depth'),
                Integer(2, 40, name='min_samples_split', ),
                Integer(1, 20, name='min_samples_leaf', ),
                Integer(1, 176, name='n_features'),
               Categorical(['sqrt', 'log2', None], name='max_features')
]

def feature_selection(df, X, y, number_of_feature):
    selector = SelectKBest(k=number_of_feature).fit(X, y)
    col_to_remove = X.columns[~selector.get_support()]
    new_df = df.drop(col_to_remove, axis=1)
    return new_df

def run_pu_learn_model(df, input_file_path, **search_space):

    df.to_json(input_file_path)

    model = DecisionTreeClassifier(**search_space,
                                   criterion="gini",
                                   class_weight="balanced",
                                   random_state=random_state)

    pul = PULearner()

    pu_stats_max = pul.cv_baggingDT(input_file_path,
                                    splits=10,
                                    repeats=10,
                                    bags=100,
                                    model=model,
                                    random_state=random_state)

    return pul, pu_stats_max

@use_named_args(search_space)
def evaluate_model(**search_space):
    print('start tuning')
    for k, v in search_space.items():
        print(k, v)

    input_df = feature_selection(all_data, X, y, number_of_feature=search_space['n_features'])

    #remove the n_features from the search space for LGBM
    search_space.pop('n_features', None)
    pul, pu_stats_max = run_pu_learn_model(input_df, input_file_path, **search_space)

    unlabelled_df = pul.df_U.copy()
    unlabelled_df['synth_score'] = pu_stats_max['prob']

    p_df = pul.df_P.copy()
    p_df['synth_score'] = pu_stats_max['prob_P_test']

    alpha = 0.1  # proportion of positives in the unlabelled set
    beta = 1  # proportion of positives in the labelled set
    class_threshold = 0.5

    y_prob_list = list(p_df['synth_score']) + list(unlabelled_df['synth_score'])
    y_test = np.zeros(len(p_df) + len(unlabelled_df))
    y_test[:len(p_df)] = 1

    roc_auc = roc_auc_score(y_test, y_prob_list)

    return -roc_auc

dataset_path = 'data'
input_file = 'set_3.json'
all_data = pd.read_json(dataset_path + '/' + input_file)
X = all_data.iloc[:, 7:-1]
y = all_data.iloc[:, -1]
input_file_path = 'data/model_3_input.json'
print('before checkpoint')

checkpoint_saver = CheckpointSaver('checkpoint/model_3.pkl')
early_stop = DeltaYStopper(0.01, 20)

print('begin hyper tuning')

result = gp_minimize(func=evaluate_model,
                     dimensions=search_space,
                     n_calls=100,
                     n_initial_points=10,
                     callback=[checkpoint_saver, early_stop],
                     random_state=random_state,
                     n_jobs=-1)

result = load('checkpoint/model_3.pkl')

def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def g_mean_score(y_true, y_pred):
    return np.sqrt(sensitivity_score(y_true, y_pred) * specificity_score(y_true, y_pred))

input_df = feature_selection(all_data, X, y, number_of_feature=result.x[3])
input_file_path = 'data/model_3_tuned_input.json'
input_df.to_json(input_file_path)

# remove the n_features from the search space for LGBM
model = DecisionTreeClassifier(
    max_depth=result.x[0],
    min_samples_split=result.x[1],
    min_samples_leaf=result.x[2],
    criterion="gini",
    class_weight="balanced",
    random_state=random_state,
    max_features=result.x[4])

pul = PULearner()

pu_stats_max = pul.cv_baggingDT(input_file_path,
                                  splits=10,
                                  repeats=10,
                                  bags=100,
                                  model=model,
                                  random_state=random_state)

unlabelled_df = pul.df_U.copy()
unlabelled_df['synth_score'] = pu_stats_max['prob']

p_df = pul.df_P.copy()
p_df['synth_score'] = pu_stats_max['prob_P_test']

y_prob_list = list(p_df['synth_score']) + list(unlabelled_df['synth_score'])
y_test = np.zeros(len(p_df) + len(unlabelled_df))
y_test[:len(p_df)] = 1

fpr, tpr, thresholds = roc_curve(y_test, y_prob_list)

g_mean_list = np.sqrt(tpr * (1 - fpr))
ix = np.nanargmax(g_mean_list)

opt_threshold = thresholds[ix]

roc_auc = roc_auc_score(y_test, y_prob_list)
print('g-mean:', np.nanmax(g_mean_list))
print('roc_auc:', roc_auc)
print(len(unlabelled_df[unlabelled_df['synth_score'] > opt_threshold]), ',',
      len(unlabelled_df[unlabelled_df['synth_score'] > opt_threshold]) / len(unlabelled_df), '%')
print(len(p_df[p_df['synth_score'] > opt_threshold]), ',', len(p_df[p_df['synth_score'] > opt_threshold]) / len(p_df),
      '%')

# Saving Results

with open('result/model_3_result.pkl', 'wb') as handle:
    pickle.dump(pu_stats_max, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_df = pd.concat([p_df, unlabelled_df])
data_df.to_csv('result/model_3_result.csv', index=False)
