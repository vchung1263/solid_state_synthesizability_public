# General
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from lightgbm import LGBMClassifier

# skopt
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, DeltaYStopper

random_state = 42
data_df = pd.read_csv('result/model_1_result.csv')

data_df.drop('PU_label', axis = 1, inplace = True)
data_df['pu_prediction'] = data_df['synth_score'].apply(lambda score: 1 if score > 0.579 else 0)

X = data_df.iloc[:, 7:-2]
y = data_df.iloc[:,-1]

model = LGBMClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state, stratify= y)

search_space = [Real(0.01, 0.1, name='lambda_l1', prior='log-uniform'),
                Real(0.1, 1, name='bagging_fraction'),
                Integer(1,20, name = 'bagging_freq'),
                Integer(2,120, name = 'num_leaves'),
                Real(0.1, 1, name = 'feature_fraction'),
                Integer(2,100, name = 'max_depth'),
                Integer(10,200, name = 'max_bin'),
                Real(0.001, 1, name = 'learning_rate', prior='log-uniform'),
                Integer(2, 100, name = 'min_data_in_leaf')]

@use_named_args(search_space)

def evaluate_model(**params):

    print(', '.join(['{}={!r}'.format(k, v) for k, v in params.items()]))

    kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
    model = LGBMClassifier(**params, device = 'gpu', random_state=random_state, verbosity=-1, is_unbalance = True, gpu_use_dp = True)

    score = cross_validate(model, X_train, y_train, cv = kf, n_jobs= 4,
                           scoring = {'roc_auc': make_scorer(roc_auc_score, needs_proba = True)}, verbose = 2)

    print(-score['test_roc_auc'].mean())
    return -score['test_roc_auc'].mean()

checkpoint_saver = CheckpointSaver('checkpoint/model_surrogate_lbgm.pkl')
early_stop = DeltaYStopper(0.01, 20)

#quick test for checkpoint, so reduced n_call first
result = gp_minimize(func = evaluate_model,
                     dimensions = search_space,
                     n_calls = 100,
                     n_initial_points = 10,
                     callback=[checkpoint_saver, early_stop],
                     random_state = random_state)