from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import recall_score
from sklearn.utils import resample
from monty.serialization import dumpfn
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

class PULearner:
    def __init__(self):
        """A machine learning model that predicts material synthesizability.
        Positive samples are experimentally synthesized materials. Unlabeled
        samples are not-yet synthesized materials.
        Hyperparameters are initialized with sensible defaults, but any newly
        trained model should have hyperparams carefully converged.
        Attributes:
            pu_stats (dict): Outputs of cv_baggingDT
            df_U (DataFrame): Unlabeled data.
            df_P (DataFrame): Positive data.
            synth_scores (list): Synthesizability scores (between 0 and 1) of
                unlabeled samples.
            labels (list): Synthesizable (1) or not (0)
            feat_importances (DataFrame): Feature importances from trained
                decision tree classifiers. Index corresponds to feature index
                in original data.
        """

        self.pu_stats = None
        self.df_U = None
        self.df_P = None
        self.synth_scores = None
        self.labels = None
        self.feat_importances = None

    def cv_baggingDT(self, pu_data, splits=10, repeats=10, bags=100, filename="", 
                     model="default", random_state=42):
        """
        Train bagged classifiers and do repeated k-fold CV.
        Default classifier is a decision tree base classifier.
        Synthesizability scores (0 = not synthesizable, 1 = already
        synthesized) are generated for an unlabeled sample by averaging
        the scores from the ensemble of classifiers that
        have not been trained on that sample.
        Args:
            pu_data (json): A file where each row describes a material.
                There MUST be a column called "PU_label" where a 1 value
                indicates a synthesized (positive) compound and a 0 value
                indicates an unlabeled compound.
            splits (int): Number of splits in k-fold CV.
            repeats (int): Number of repeated k-fold CV.
            bags (int): Number of bags in bootstrap aggregation.
            filename (string): Save model training results to file with
                filename ending in .json or .pkl.
            model ("default", sklearn model): Classifier used for training.
                Default model is sklearn decision tree classifier.
            random_state (int, RandomState instance, or None): 
                Controls the randomness of k-fold validation and classifiers
        Returns:
            pu_stats (dict): Metrics and outputs of PU learning model
                training.
        """

        print("Start PU Learning.")

        np.random.seed(random_state)

        # Preprocess of data and set attributes
        df = pd.read_json(pu_data)
        df_P, df_U, X_P, X_U = self._process_pu_data(df)
        self.df_P = df_P
        self.df_U = df_U

        # Split data into training and test splits for k-fold CV
        kfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=random_state)

        # Scores for PU learning (tpr = True Positive Rate)
        scores = []
        tprs = []

        # Predicted synthesis probability of CVed P and U sets, prob_P_test is for validation later
        prob_P = np.ones(shape=(X_P.shape[0], splits * repeats))
        prob_U = -np.ones(shape=(X_U.shape[0], splits * repeats))
        prob_P_test = [[] for i in range(X_P.shape[0])]

        # Feature importance
        feat_rank = np.zeros(shape=(X_P.shape[1], splits * repeats))

        # index of repeated k splits
        idsp = 0 

        # Loop over P and U training/test samples
        for (ptrain, ptest), (utrain, utest) in zip(kfold.split(X_P), kfold.split(X_U)):

            # Number of P and U training samples
            N_ptrain = X_P[ptrain].shape[0]
            N_utrain = X_U[utrain].shape[0]

            train_label = np.zeros(shape=(2 * N_ptrain,))
            # Synthesized (positive)
            train_label[:N_ptrain] = 1.0

            # Out of bag samples
            n_oob = np.zeros(shape=(N_utrain,))
            f_oob = np.zeros(shape=(N_utrain, 2))

            # Sums of probabilities of test sets
            f_ptest = np.zeros(shape=(X_P[ptest].shape[0], 2))
            f_utest = np.zeros(shape=(X_U[utest].shape[0], 2))

            # Bootstrap resampling for each bag
            for i in range(bags):
                bootstrap_sample = np.random.choice(
                    np.arange(N_utrain), replace=True, size=N_ptrain
                )

                # Positive samples and bootstrapped unlabeled samples
                data_bootstrap = np.concatenate(
                    (X_P[ptrain], X_U[bootstrap_sample, :]), axis=0
                )

                # Train decision tree classifier for default
                if model == "default":
                    model = DecisionTreeClassifier(
                        max_depth=None,
                        max_features=None,
                        criterion="gini",
                        class_weight="balanced",
                    )

                model.fit(data_bootstrap, train_label)

                # Index for the oob samples
                idx_oob = sorted(
                    set(range(N_utrain)) - set(np.unique(bootstrap_sample))
                )

                # Transductive learning on oob samples
                f_oob[idx_oob] += model.predict_proba(X_U[utrain][idx_oob])
                n_oob[idx_oob] += 1
                f_ptest += model.predict_proba(X_P[ptest])
                f_utest += model.predict_proba(X_U[utest])
                feat_rank[:, idsp] = model.feature_importances_

            # Predicted synthesis probabilities of unlabeled samples
            predict_utrain = f_oob[:, 1] / n_oob

            # Predicted probabilities for P and U test sets
            predict_ptest = f_ptest[:, 1] / bags
            predict_utest = f_utest[:, 1] / bags

            # Find predicted positives
            true_pos = predict_ptest[np.where(predict_ptest > 0.5)].shape[0]
            u_pos = predict_utest[np.where(predict_utest > 0.5)].shape[0]

            N_ptest = X_P[ptest].shape[0]
            N_utest = X_U[utest].shape[0]

            # Predicted positive ratio in test set
            p_pred_pos = (true_pos + u_pos) / (N_ptest + N_utest) + 0.0001

            # Compute PU recall (TPR) and score metrics
            recall = true_pos / N_ptest
            score = recall ** 2 / p_pred_pos
            scores.append(score)
            tprs.append(recall)

            # Predicted probabilities
            prob_P[ptest, idsp] = predict_ptest
            for ptest_index, ptest_prob in zip(ptest, predict_ptest):
                prob_P_test[ptest_index].append(ptest_prob)
            prob_U[utrain, idsp] = predict_utrain
            prob_U[utest, idsp] = predict_utest
            idsp += 1

            # Progress update
            if (idsp + 1) % splits == 0:
                tpr_tmp = np.asarray(tprs[-splits - 1: -1])
                print(
                    "Performed Repeated "
                    + str(splits)
                    + "-fold: "
                    + str(idsp // splits + 1)
                    + " out of "
                    + str(repeats)
                )
                print(
                    "True Positive Rate: %0.2f (+/- %0.2f)"
                    % (tpr_tmp.mean(), tpr_tmp.std() * 2)
                )

        tprs = np.asarray(tprs)
        scores = np.asarray(scores)

        # Metrics for each model in the k-folds
        label_U_rp = np.zeros(shape=(X_U.shape[0], repeats), dtype=int)
        prob_U_rp = np.zeros(shape=(X_U.shape[0], repeats))
        feat_rank_rp = np.zeros(shape=(X_U.shape[1], repeats))
        tpr_rp = np.zeros(shape=(repeats,))
        scores_rp = np.zeros(shape=(repeats,))
        labels = np.zeros(shape=(X_U.shape[0],))

        for i in range(repeats):
            prob_U_rp[:, i] = prob_U[:, i * splits: (i + 1) * splits].mean(axis=1)
            feat_rank_rp[:, i] = feat_rank[:, i * splits: (i + 1) * splits].mean(
                axis=1
            )
            tpr_rp[i] = tprs[i * splits: (i + 1) * splits].mean()
            scores_rp[i] = scores[i * splits: (i + 1) * splits].mean()

        label_U_rp[np.where(prob_U_rp > 0.5)] = 1
        prob = prob_U_rp.mean(axis=1)
        prob_P_test = np.array(prob_P_test).mean(axis=1)
        labels[np.where(prob > 0.5)] = 1

        # Get confidence interval of TPR for each kfold
        tpr_low, tpr_up = self.bootstrapCI(tpr_rp)
        scores_low, scores_up = self.bootstrapCI(scores_rp)

        # PU learning metrics
        metrics = np.asarray(
            [tpr_rp.mean(), tpr_low, tpr_up, scores_rp.mean(), scores_low, scores_up]
        )

        print("Accuracy: %0.2f" % (tpr_rp.mean()))
        print("95%% confidence interval: [%0.2f, %0.2f]" % (tpr_low, tpr_up))

        # Metrics and results from training / testing
        pu_stats = {
            "prob": prob,
            "labels": labels,
            "metrics": metrics,
            "prob_rp": prob_U_rp,
            "label_rp": label_U_rp,
            "tpr_rp": tpr_rp,
            "scores_rp": scores_rp,
            "feat_rank_rp": feat_rank_rp,
            'prob_P_test': prob_P_test,
        }

        # Save results
        if filename:
            if filename.endswith(".json"):
                dumpfn(pu_stats, filename)
            if filename.endswith(".pkl"):
                with open(filename, "wb") as file:
                    pickle.dump(pu_stats, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.pu_stats = pu_stats
        return pu_stats

    def bootstrapCI(self, data, ci=95, ns=10000):
        """Compute confidence interval of the TPR.
        Args:
            data (array): Array of TPRs for each kfold.
            ci (int): Confidence interval.
            ns (int): Number of bootstrap resamplings.
        Returns:
            lower (float): Lower endpoint of CI.
            upper (float): Upper endpoint of CI.

        """

        bs_rsample = []
        for _ in range(ns):
            rsample = resample(data, n_samples=len(data))
            bs_rsample.append(np.mean(rsample))

        bs_rsample = np.asarray(bs_rsample)
        lower = np.percentile(bs_rsample, (100 - ci) / 2)
        upper = np.percentile(bs_rsample, ci + (100 - ci) / 2)

        return lower, upper

    @staticmethod
    def _process_pu_data(data):
        """Utility method for processing input data.
        Args:
            data (DataFrame): Data with positive and unlabeled samples.
        Returns:
            X_P (array): Positive sample set.
            X_U (array): Unlabeled sample set.
        """

        df_P = data.query("PU_label == 1")
        df_U = data.query("PU_label == 0") 

        # Chop off PU label and drop non-numeric columns for sklearn
        X_P = np.asarray(df_P.drop(columns=["PU_label"])._get_numeric_data())
        X_U = np.asarray(df_U.drop(columns=["PU_label"])._get_numeric_data())

        return df_P, df_U, X_P, X_U


class PNULearner:
    def __init__(self):
        """A machine learning model that predicts material synthesizability.
        Positive samples are experimentally synthesized materials. Unlabeled
        samples are not-yet synthesized materials. Negaive samples are non-
        synthesizable materials.
        Hyperparameters are initialized with sensible defaults, but any newly
        trained model should have hyperparams carefully converged.
        Attributes:
            pu_stats (dict): Outputs of cv_baggingDT
            df_U (DataFrame): Unlabeled data.
            df_N (DataFrame): Negative data.
            df_P (DataFrame): Positive data.
            synth_scores (list): Synthesizability scores (between 0 and 1) of
                unlabeled samples.
            labels (list): Solid-state synthesizable (1) or not (0)
            feat_importances (DataFrame): Feature importances from trained
                decision tree classifiers. Index corresponds to feature index
                in original data.
        """

        self.pu_stats = None
        self.df_U = None
        self.df_N = None
        self.df_P = None
        self.synth_scores = None
        self.labels = None
        self.feat_importances = None

    def cv_baggingDT(self, pu_data, splits=10, repeats=10, bags=100, filename="", 
                    user_metrics=[], model='default', random_state=42):
        """
        Train bagged classifiers and do repeated k-fold CV.
        Default classifier is a decision tree base classifier.
        Synthesizability scores (0 = not synthesizable, 1 = already
        synthesized) are generated for an unlabeled sample by averaging
        the scores from the ensemble of classifiers that
        have not been trained on that sample.
        Args:
            pu_data (json): A file where each row describes a material.
                There MUST be a column called "PU_label" where a 1 value
                indicates a synthesized (positive) compound and a 0 value
                indicates an unlabeled compound.
            splits (int): Number of splits in k-fold CV.
            repeats (int): Number of repeated k-fold CV.
            bags (int): Number of bags in bootstrap aggregation.
            filename (string): Save model training results to file with
                filename ending in .json or .pkl.
            usermetric (array of sklearn scorer): An array of metrics that
                takes (y_true, y_pred) as input and return a model score,
                where y_true and y_pred are 1-d arrays of ground truth and
                predicted labels, respectively.
            model ("default", sklearn model): Classifier used for training.
                Default model is sklearn decision tree classifier.
            random_state (int, RandomState instance, or None): 
                Controls the randomness of k-fold validation and classifiers
        Returns:
            pu_stats (dict): Metrics and outputs of PU learning model
                training.
        """

        print("Start PU Learning.")

        np.random.seed(random_state)

        # Preprocess data and set attributes
        df = pd.read_json(pu_data)
        df_P, df_N, df_U, X_P, X_N, X_U = self._process_pu_data(df)
        self.df_P = df_P
        self.df_N = df_N
        self.df_U = df_U

        # Split data into training and test splits for k-fold CV
        kfold = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=random_state)

        # Scores for PU learning (tpr = True Positive Rate)
        scores = []
        tprs = []
        user_metric_result_list = []
        for _ in user_metrics:
            user_metric_result_list.append([])

        # Predicted synthesis probability of CVed P, N and U sets, prob_P_test and prob_N_test is for later use
        prob_P = np.ones(shape=(X_P.shape[0], splits * repeats))
        prob_N = np.zeros(shape=(X_N.shape[0], splits * repeats))
        prob_U = -np.ones(shape=(X_U.shape[0], splits * repeats))
        prob_P_test = [[] for i in range(X_P.shape[0])]
        prob_N_test = [[] for i in range(X_N.shape[0])]

        # Feature importance
        feat_rank = np.zeros(shape=(X_P.shape[1], splits * repeats))

        idsp = 0  # index of repeated k splits

        # Loop over P and U training/test samples
        for (ptrain, ptest), (ntrain, ntest), (utrain, utest) in zip(kfold.split(X_P), kfold.split(X_N),
                                                                     kfold.split(X_U)):

            # Number of P and U training samples
            N_ptrain = X_P[ptrain].shape[0]
            N_ntrain = X_N[ntrain].shape[0]
            N_utrain = X_U[utrain].shape[0]

            # 2 * N_ptrain will be the total number of data in training
            train_label = np.zeros(shape=(2 * N_ptrain,))
            train_label[:N_ptrain] = 1.0  # Synthesized (positive)

            # Out of bag samples
            # n_oob represents the number of time of the oob sample has been predicted in each k fold
            # f_oob holds the prediction of each oob sample as a tuples (predicted negative vs predicted positive in each k fold)
            n_oob = np.zeros(shape=(N_utrain,))
            f_oob = np.zeros(shape=(N_utrain, 2))

            # Sums of probabilities of test sets
            f_ptest = np.zeros(shape=(X_P[ptest].shape[0], 2))
            f_ntest = np.zeros(shape=(X_N[ntest].shape[0], 2))
            f_utest = np.zeros(shape=(X_U[utest].shape[0], 2))

            # Bootstrap resampling for each bag
            # changed size from K to K-N_ntrain to take into account the negative data used in training
            for i in range(bags):
                bootstrap_sample = np.random.choice(
                    np.arange(N_utrain), replace=True, size=N_ptrain - N_ntrain
                )

                # Positive, negative and bootstrapped unlabeled samples
                data_bootstrap = np.concatenate(
                    (X_P[ptrain], X_N[ntrain], X_U[bootstrap_sample, :]), axis=0
                )

                # Train decision tree classifier
                if model == 'default':
                    model = DecisionTreeClassifier(
                        max_depth=None,
                        max_features=None,
                        criterion="gini",
                        class_weight="balanced",
                    )

                model.fit(data_bootstrap, train_label)

                # Index for the oob samples
                idx_oob = sorted(
                    set(range(N_utrain)) - set(np.unique(bootstrap_sample))
                )

                # Transductive learning on oob samples
                f_oob[idx_oob] += model.predict_proba(X_U[utrain][idx_oob])
                n_oob[idx_oob] += 1
                f_ptest += model.predict_proba(X_P[ptest])
                f_ntest += model.predict_proba(X_N[ntest])
                f_utest += model.predict_proba(X_U[utest])
                feat_rank[:, idsp] = model.feature_importances_

            # Predicted synthesis probabilities of unlabeled samples
            predict_utrain = f_oob[:, 1] / n_oob

            # Predicted probabilities for P, N and U test sets
            predict_ptest = f_ptest[:, 1] / bags
            predict_ntest = f_ntest[:, 1] / bags
            predict_utest = f_utest[:, 1] / bags

            # Find predicted positives
            true_pos = predict_ptest[np.where(predict_ptest >= 0.5)].shape[0]
            false_pos = predict_ntest[np.where(predict_ntest >= 0.5)].shape[0]
            u_pos = predict_utest[np.where(predict_utest > 0.5)].shape[0]

            N_ptest = X_P[ptest].shape[0]
            N_ntest = X_N[ntest].shape[0]
            N_utest = X_U[utest].shape[0]

            # used to predict other metrics
            y_true = np.concatenate([np.ones(len(predict_ptest)), np.zeros(len(predict_ntest))])
            y_pred_prob = np.concatenate([predict_ptest, predict_ntest])
            y_pred = y_pred_prob.copy()
            y_pred[np.where(y_pred >= 0.5)] = 1
            y_pred[np.where(y_pred < 0.5)] = 0

            # Predicted positive ratio in test set
            p_pred_pos = (true_pos + u_pos + false_pos) / (N_ptest + N_ntest + N_utest) + 0.0001

            # Compute PU recall (TPR) and score metrics
            recall = recall_score(y_true, y_pred)
            score = recall ** 2 / p_pred_pos
            scores.append(score)
            tprs.append(recall)

            # custom metric
            for index, m in enumerate(user_metrics):
                result = m(y_true, y_pred)
                user_metric_result_list[index].append(result)

            # Predicted probabilities
            prob_P[ptest, idsp] = predict_ptest
            prob_N[ntest, idsp] = predict_ntest
            prob_U[utrain, idsp] = predict_utrain
            prob_U[utest, idsp] = predict_utest
            for ptest_index, ptest_prob in zip(ptest, predict_ptest):
                prob_P_test[ptest_index].append(ptest_prob)
            for ntest_index, ntest_prob in zip(ntest, predict_ntest):
                prob_N_test[ntest_index].append(ntest_prob)
            idsp += 1

            # Progress update
            if (idsp + 1) % splits == 0:
                tpr_tmp = np.asarray(tprs[-splits - 1: -1])

                print(
                    "Performed Repeated "
                    + str(splits)
                    + "-fold: "
                    + str(idsp // splits + 1)
                    + " out of "
                    + str(repeats)
                )
                print(
                    "True Positive Rate: %0.2f (+/- %0.2f)"
                    % (tpr_tmp.mean(), tpr_tmp.std() * 2)
                )
                for index, res in enumerate(user_metric_result_list):
                    result_tmp = np.asarray(res[-splits - 1: -1])
                    print(
                        "User Metric %s: %0.2f (+/- %0.2f)"
                        % (index, result_tmp.mean(), result_tmp.std() * 2)
                    )

        tprs = np.asarray(tprs)
        scores = np.asarray(scores)

        for index, res in enumerate(user_metric_result_list):
            user_metric_result_list[index] = np.asarray(res)

        # Metrics for each model in the k-folds
        label_U_rp = np.zeros(shape=(X_U.shape[0], repeats), dtype=int)
        prob_U_rp = np.zeros(shape=(X_U.shape[0], repeats))
        feat_rank_rp = np.zeros(shape=(X_U.shape[1], repeats))
        tpr_rp = np.zeros(shape=(repeats,))
        scores_rp = np.zeros(shape=(repeats,))
        labels = np.zeros(shape=(X_U.shape[0],))
        user_metric_rp = []

        for _ in user_metric_result_list:
            user_metric_rp.append(np.zeros(shape=(repeats,)))

        for i in range(repeats):
            prob_U_rp[:, i] = prob_U[:, i * splits: (i + 1) * splits].mean(axis=1)
            feat_rank_rp[:, i] = feat_rank[:, i * splits: (i + 1) * splits].mean(
                axis=1
            )
            tpr_rp[i] = tprs[i * splits: (i + 1) * splits].mean()
            scores_rp[i] = scores[i * splits: (i + 1) * splits].mean()
            for index, user_metric_result in enumerate(user_metric_result_list):
                user_metric_rp[index][i] = user_metric_result[i * splits: (i + 1) * splits].mean()

        label_U_rp[np.where(prob_U_rp > 0.5)] = 1
        prob = prob_U_rp.mean(axis=1)
        labels[np.where(prob > 0.5)] = 1

        # Get confidence interval of TPR for each kfold
        tpr_low, tpr_up = self.bootstrapCI(tpr_rp)
        scores_low, scores_up = self.bootstrapCI(scores_rp)
        prob_P_test = np.array(prob_P_test).mean(axis=1)
        prob_N_test = np.array(prob_N_test).mean(axis=1)
        user_metric_low_list, user_metric_high_list = [], []

        for user_metric_result_rp in user_metric_rp:
            user_metric_low, user_metric_high = self.bootstrapCI(user_metric_result_rp)
            user_metric_low_list.append(user_metric_low)
            user_metric_high_list.append(user_metric_high)

        # PU learning metrics
        metrics = np.asarray(
            [tpr_rp.mean(), tpr_low, tpr_up, scores_rp.mean(), scores_low, scores_up,
             [i.mean() for i in user_metric_rp], user_metric_low_list, user_metric_high_list]
        )

        print("Accuracy (Recall): %0.2f" % (tpr_rp.mean()))
        print("95%% confidence interval: [%0.2f, %0.2f]" % (tpr_low, tpr_up))

        # Metrics and results from training / testing
        pu_stats = {
            "prob": prob,
            "labels": labels,
            "metrics": metrics,
            "prob_rp": prob_U_rp,
            "label_rp": label_U_rp,
            "tpr_rp": tpr_rp,
            "scores_rp": scores_rp,
            "feat_rank_rp": feat_rank_rp,
            'prob_P_test': prob_P_test,
            'prob_N_test': prob_N_test,
        }

        # Save results
        if filename:
            if filename.endswith(".json"):
                dumpfn(pu_stats, filename)
            if filename.endswith(".pkl"):
                with open(filename, "wb") as file:
                    pickle.dump(pu_stats, file, protocol=pickle.HIGHEST_PROTOCOL)

        self.pu_stats = pu_stats
        return pu_stats

    def bootstrapCI(self, data, ci=95, ns=10000):
        """Compute confidence interval of the TPR.
        Args:
            data (array): Array of TPRs for each kfold.
            ci (int): Confidence interval.
            ns (int): Number of bootstrap resamplings.
        Returns:
            lower (float): Lower endpoint of CI.
            upper (float): Upper endpoint of CI.

        """

        bs_rsample = []
        for _ in range(ns):
            rsample = resample(data, n_samples=len(data))
            bs_rsample.append(np.mean(rsample))

        bs_rsample = np.asarray(bs_rsample)
        lower = np.percentile(bs_rsample, (100 - ci) / 2)
        upper = np.percentile(bs_rsample, ci + (100 - ci) / 2)

        return lower, upper

    @staticmethod
    def _process_pu_data(data):
        """Utility method for processing input data.
        Args:
            data (DataFrame): Data with positive and unlabeled samples.
        Returns:
            X_P (array): Positive sample set.
            X_N (array): Negative sample set.
            X_U (array): Unlabeled sample set.
        """

        df_P = data.query("PU_label == 1")  # Positive value is 1
        df_N = data.query("PU_label == 0")  # Negative value is 0
        df_U = data.query("PU_label == -1")  # Unlabeled value is -1

        # Chop off PU label and drop non-numeric columns for sklearn
        X_P = np.asarray(df_P.drop(columns=["PU_label"])._get_numeric_data())
        X_N = np.asarray(df_N.drop(columns=["PU_label"])._get_numeric_data())
        X_U = np.asarray(df_U.drop(columns=["PU_label"])._get_numeric_data())

        return df_P, df_N, df_U, X_P, X_N, X_U
