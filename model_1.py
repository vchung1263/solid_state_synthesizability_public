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

    n_df = unlabelled_df[unlabelled_df['sss_synthesized_label'] == 'other']

    p_df = pul.df_P.copy()
    p_df['synth_score'] = pu_stats_max['prob_P_test']

    y_prob_list = list(p_df['synth_score']) + list(n_df['synth_score'])
    y_test = np.zeros(len(p_df) + len(n_df))
    y_test[:len(p_df)] = 1

    roc_auc = roc_auc_score(y_test, y_prob_list)

    return -roc_auc

dataset_path = 'data'
input_file = 'set_1.json'
all_data = pd.read_json(dataset_path + '/' + input_file)
X = all_data.iloc[:, 7:-1]
y = all_data.iloc[:, -1]
input_file_path = 'data/model_1_input.json'

checkpoint_saver = CheckpointSaver('checkpoint/model_1.pkl')
early_stop = DeltaYStopper(0.01, 20)

result = gp_minimize(func=evaluate_model,
                     dimensions=search_space,
                     n_calls=100,
                     n_initial_points=10,
                     callback=[checkpoint_saver, early_stop],
                     random_state=random_state,
                     n_jobs=-1)

result = load('checkpoint/model_1.pkl')

input_df = feature_selection(all_data, X, y, number_of_feature=result.x[3])
input_file_path = 'data/model_1_tuned_input.json'
input_df.to_json(input_file_path)

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
unlabelled_df = unlabelled_df.sort_values(by='synth_score', ascending=False)
unlabelled_df.head()

hypo_df = unlabelled_df[unlabelled_df['sss_synthesized_label'] == 'hypo']
n_df = unlabelled_df[unlabelled_df['sss_synthesized_label'] == 'other']

p_df = pul.df_P.copy()
p_df['synth_score'] = pu_stats_max['prob_P_test']

y_prob_list = list(p_df['synth_score']) + list(n_df['synth_score'])
y_test = np.zeros(len(p_df) + len(n_df))
y_test[:len(p_df)] = 1

# Saving Results

with open('result/model_1_result.pkl', 'wb') as handle:
    pickle.dump(pu_stats_max, handle, protocol=pickle.HIGHEST_PROTOCOL)

data_df = pd.concat([p_df, n_df, hypo_df])
data_df.to_csv('result/model_1_result.csv', index=False)