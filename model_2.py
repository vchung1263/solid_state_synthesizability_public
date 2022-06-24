# General
import numpy as np
import pandas as pd
import pickle
from pu_learn_model import PNULearner
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
    new_X = X.drop(col_to_remove, axis=1)
    return new_df

def run_pnu_learn_model(df, input_file_path, **search_space):

    df.to_json(input_file_path)

    model = DecisionTreeClassifier(**search_space,
                                   criterion="gini",
                                   class_weight="balanced",
                                   random_state=random_state)

    pnul = PNULearner()

    pnu_stats_max = pnul.cv_baggingDT(input_file_path,
                                    splits=10,
                                    repeats=10,
                                    bags=100,
                                    model=model,
                                    random_state=random_state)

    return pnul, pnu_stats_max

@use_named_args(search_space)
def evaluate_model(**search_space):
    print('start tuning')
    for k, v in search_space.items():
        print(k, v)

    input_df = feature_selection(all_data, X, y, number_of_feature=search_space['n_features'])

    #remove the n_features from the search space for LGBM
    search_space.pop('n_features', None)
    pnul, pnu_stats_max = run_pnu_learn_model(input_df, input_file_path, **search_space)

    p_df = pnul.df_P.copy()
    n_df = pnul.df_N.copy()
    p_df['synth_score'] = pnu_stats_max['prob_P_test']
    n_df['synth_score'] = pnu_stats_max['prob_N_test']

    unlabelled_df = pnul.df_U.copy()
    unlabelled_df['synth_score'] = pnu_stats_max['prob']

    y_prob_list = list(p_df['synth_score']) + list(n_df['synth_score'])
    y_test = np.zeros(len(p_df) + len(n_df))
    y_test[:len(p_df)] = 1

    roc_auc = roc_auc_score(y_test, y_prob_list)

    print('roc score:', -roc_auc)

    # want to max tpr, so negative it
    return -roc_auc

print('before checkpoint')

checkpoint_saver = CheckpointSaver('checkpoint/model_2.pkl')
early_stop = DeltaYStopper(0.01, 20)

print('begin hyper tuning')

dataset_path = 'data'
input_file = 'set_2.json'
all_data = pd.read_json(dataset_path + '/' + input_file)
X = all_data.iloc[:, 7:-1]
y = all_data.iloc[:, -1]
input_file_path = 'data/model_2_input.json'

result = gp_minimize(func=evaluate_model,
                     dimensions=search_space,
                     n_calls=100,
                     n_initial_points=10,
                     callback=[checkpoint_saver, early_stop],
                     random_state=random_state,
                     n_jobs=-1)

result = load('checkpoint/model_2.pkl')

#the metrics
def sensitivity_score(y_true, y_pred):
    return recall_score(y_true, y_pred)

def specificity_score(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def g_mean_score(y_true, y_pred):
    return np.sqrt(sensitivity_score(y_true, y_pred) * specificity_score(y_true, y_pred))

input_df = feature_selection(all_data, X, y, number_of_feature=result.x[3])
input_file_path = 'data/model_2_tuned_input.json'
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

pnul = PNULearner()

pnu_stats_max = pnul.cv_baggingDT(input_file_path,
                                  splits=10,
                                  repeats=10,
                                  bags=100,
                                  model=model,
                                  user_metrics=[g_mean_score],
                                  random_state=random_state)

unlabelled_df = pnul.df_U.copy()
unlabelled_df['synth_score'] = pnu_stats_max['prob']
unlabelled_df = unlabelled_df.sort_values(by='synth_score', ascending=False)

hypo_df = unlabelled_df[unlabelled_df['sss_synthesized_label'] == 'hypo']

p_df = pnul.df_P.copy()
n_df = pnul.df_N.copy()
p_df['synth_score'] = pnu_stats_max['prob_P_test']
n_df['synth_score'] = pnu_stats_max['prob_N_test']

y_prob_list = list(p_df['synth_score']) + list(n_df['synth_score'])
y_test = np.zeros(len(p_df) + len(n_df))
y_test[:len(p_df)] = 1

fpr, tpr, thresholds = roc_curve(y_test, y_prob_list)
roc_auc = roc_auc_score(y_test, y_prob_list)

g_mean_list = np.sqrt(tpr * (1 - fpr))
ix = np.nanargmax(g_mean_list)

opt_threshold = thresholds[ix]
print('g-mean:', np.nanmax(g_mean_list))
print('roc_auc:', roc_auc)

print(len(hypo_df[hypo_df['synth_score'] > opt_threshold]), ',',
      len(hypo_df[hypo_df['synth_score'] > opt_threshold]) / len(hypo_df), '%')
print(len(n_df[n_df['synth_score'] > opt_threshold]), ',', len(n_df[n_df['synth_score'] > opt_threshold]) / len(n_df),
      '%')
print(len(p_df[p_df['synth_score'] > opt_threshold]), ',', len(p_df[p_df['synth_score'] > opt_threshold]) / len(p_df),
      '%')

# Saving Results

with open('result/model_2_result.pkl', 'wb') as handle:
    pickle.dump(pnu_stats_max, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_df = pd.concat([p_df, n_df, hypo_df])
data_df.to_csv('result/model_2_result.csv', index=False)