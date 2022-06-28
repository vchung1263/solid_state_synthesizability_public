# General
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score , make_scorer, roc_curve
from lightgbm import LGBMClassifier

# skopt
import skopt
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver

data_df = pd.read_json('data/set_3.json')

p_df = data_df[data_df['synthesized_label'] == 'y']
hypo_df = data_df[data_df['synthesized_label'] == 'n']

X_p = p_df[p_df['sss_synthesized_label'] == 'sss'].iloc[:,7:-1]
X_n = p_df[p_df['sss_synthesized_label'] == 'other'].iloc[:,7:-1]
X_hypo = hypo_df.iloc[:,7:-1]
y_p = np.ones(len(X_p))
y_n = np.zeros(len(X_n))
X = pd.concat([X_p, X_n])
y = np.concatenate((y_p,y_n), axis = None)

random_state = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= random_state, test_size = 0.2, stratify= y)

search_space = [Real(0.01, 0.1, name='lambda_l1', prior='log-uniform'),
                Real(0.1, 1, name='bagging_fraction'),
                Integer(1,20, name = 'bagging_freq'),
                Integer(2,120, name = 'num_leaves'),
                Real(0.1, 1, name = 'feature_fraction'),
                Integer(2,100, name = 'max_depth'),
                Integer(10,200, name = 'max_bin'),
                Real(0.001, 1, name = 'learning_rate', prior='log-uniform'),
                Integer(2, 100, name = 'min_data_in_leaf'),
                 Integer(1, 176, name='n_features')]

@use_named_args(search_space)

def evaluate_model(**search_space):

    print('start tuning')
    for k, v in search_space.items():
        print(k, v)
    X_temp = X_train.copy()
    y_temp = y_train.copy()

    selector = SelectKBest(k = search_space['n_features']).fit(X_temp, y_temp)
    search_space.pop('n_features', None)

    col_to_remove = X_temp.columns[~selector.get_support()]
    X_temp = X_temp.drop(col_to_remove, axis = 1)

    kf = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
    model = LGBMClassifier(**search_space,
                           device = 'gpu',
                           seed=random_state,
                           gpu_use_dp=True,
                           verbose = -1)

    score = cross_validate(model, X_temp, y_temp, cv = kf, n_jobs= 4,
                           scoring = {'roc_auc': make_scorer(roc_auc_score, needs_proba= True)})

    # want to max tpr, so negative it
    print(-score['test_roc_auc'].mean())
    return -score['test_roc_auc'].mean()

checkpoint_saver = CheckpointSaver('checkpoint/basemodel_lgbm.pkl')
from skopt.callbacks import DeltaYStopper

early_stop = DeltaYStopper(0.01, 20)

#quick test for checkpoint, so reduced n_call first
result = gp_minimize(func = evaluate_model,
                     dimensions = search_space,
                     n_calls = 100,
                     n_initial_points = 10,
                     callback=[checkpoint_saver,early_stop],
                     random_state = random_state
                     )

result = skopt.load('checkpoint/basemodel_lgbm.pkl')

X_train_filered = X_train.copy()

selector = SelectKBest(k=result.x[9]).fit(X_train_filered, y_train)

col_to_remove = X_train_filered.columns[~selector.get_support()]
X_train_filered = X_train_filered.drop(col_to_remove, axis=1)
X_hypo_filtered = X_hypo.drop(col_to_remove, axis = 1)
X_test_filtered = X_test.drop(col_to_remove, axis = 1)

hyper_model = LGBMClassifier(
    lambda_l1=result.x[0],
    bagging_fraction=result.x[1],
    bagging_freq=result.x[2],
    num_leaves=result.x[3],
    feature_fraction=result.x[4],
    max_depth=result.x[5],
    max_bin=result.x[6],
    learning_rate=result.x[7],
    min_data_in_leaf=result.x[8],
    device = 'gpu',
    random_state=random_state).fit(X_train_filered, y_train)

y_pred_prob = hyper_model.predict_proba(X_test_filtered)[:,1]
y_hypo_pred_proba = hyper_model.predict_proba(X_hypo_filtered)[:,1]
hypo_df['synth_score'] = y_hypo_pred_proba

X_test_filtered['synth_score'] = y_pred_prob
X_test_filtered['true_label'] = y_test

hypo_df.to_csv('result/model_base_hypo.csv', index = False)
X_test_filtered.to_csv('result/model_base_test.csv', index = False)