import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier as EXT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.utils import shuffle

warnings.filterwarnings("ignore")

MSG_START = "------------------------>SSL RATIO {} {}<-------------------"
MSG_END = "---------------------------- FINISHED -----------------------"


def download_dataset(dataset):
    dataset_dir = "../datasets"
    if dataset == 1:
        data = pd.read_csv(f"{dataset_dir}/voice_numeric.csv")
    elif dataset == 2:
        data = pd.read_csv(f"{dataset_dir}/ANAD.csv")
    else:
        data = pd.read_csv(f"{dataset_dir}/FCTH_MFCC_Speakers8_CHAINS.csv")

    data = shuffle(data, random_state=1)

    X = data.astype("float64")
    y = data.iloc[:, -1]
    print("given dataset: ", X.shape, y.shape)
    return (X, y)


def split(train_size):
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_full, y_train_full, X_test, y_test)


def split_alssl(X, y, repeats, testsize):

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=testsize, random_state=23)
    x_tr, y_tr, x_ts, y_ts = [], [], [], []
    for train_index, test_index in sss.split(X, y):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        x_tr.append(X_train)
        y_tr.append(y_train)
        x_ts.append(X_test)
        y_ts.append(y_test)
    return x_tr, y_tr, x_ts, y_ts


def split_alssl_lists(X, y, repeats, testsize):

    sss = StratifiedShuffleSplit(n_splits=repeats, test_size=testsize, random_state=23)
    x_tr, y_tr, x_ts, y_ts = [], [], [], []
    for kk in range(0, len(X)):
        for train_index, test_index in sss.split(X[kk], y[kk]):

            X_train, X_test = (
                X[kk].loc[X[kk].index[train_index]],
                X[kk].loc[X[kk].index[test_index]],
            )
            y_train, y_test = (
                y[kk].loc[y[kk].index[train_index]],
                y[kk].loc[y[kk].index[test_index]],
            )
            x_tr.append(X_train)
            y_tr.append(y_train)
            x_ts.append(X_test)
            y_ts.append(y_test)
    return x_tr, y_tr, x_ts, y_ts, train_index, test_index


def remove_class(X_list):
    l = []
    for i in X_list:
        features = i.iloc[:, :-1]
        l.append(features)
    return l


class BaseModel(object):
    def __init__(self):
        pass

    def fit_predict(self):
        pass


class RfModel(BaseModel):

    model_type = "Random Forest"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        self.classifier = RF(n_estimators=100, class_weight=c_weight, random_state=23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class ExtModel(BaseModel):

    model_type = "Extra Trees"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = EXT(n_estimators=100, class_weight=c_weight, random_state=23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class NBModel(BaseModel):

    model_type = "NB"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = NB()
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class KNNModel(BaseModel):

    model_type = "KNN"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = KNN()
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class MLPModel(BaseModel):

    model_type = "MLP"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = MLP(hidden_layer_sizes=(100,), random_state=23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class MLPModel_2layers(BaseModel):

    model_type = "MLP2"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = MLP(hidden_layer_sizes=(100, 50), random_state=23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class MLPModel_3layers(BaseModel):

    model_type = "MLP3"

    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):

        self.classifier = MLP(hidden_layer_sizes=(200, 100, 50), random_state=23)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)


class TrainModel:
    def __init__(self, model_object):
        self.accuracies = []
        self.model_object = model_object()
        self.precision = []
        self.recall = []
        self.fscore = []

    def print_model_type(self):
        print(self.model_object.model_type)

    def train(self, X_train, y_train, X_val, X_test, c_weight):

        t0 = time.time()
        (
            X_train,
            X_val,
            X_test,
            self.val_y_predicted,
            self.test_y_predicted,
        ) = self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_test)

    def get_test_accuracy(self, i, y_test, message_AL_SSL):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_test, self.test_y_predicted, average="weighted"
        )
        self.accuracies.append(np.round(classif_rate, 3))
        self.precision.append(np.round(prec * 100, 3))
        self.recall.append(np.round(rec * 100, 3))
        self.fscore.append(np.round(f1 * 100, 3))


class BaseSelectionFunction(object):
    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):

        selection = np.random.choice(
            probas_val.shape[0], initial_labeled_samples, replace=False
        )
        return selection


class EntropySelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection


class MarginSamplingSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection


class MinStdSelection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, initial_labeled_samples):
        std = np.std(probas_val * 100, axis=1)
        selection = std.argsort()[:initial_labeled_samples]
        selection = selection.astype("int64")

        return selection


class NoActiveLearningSelection(BaseSelectionFunction):
    pass


class SSLselection(BaseSelectionFunction):
    @staticmethod
    def select(probas_val, toquery):
        maxInRows = np.amax(probas_val, axis=1)  # Maximum per row/instance
        maxInPos = []
        for i in range(0, probas_val.shape[0]):
            maxInPos.append(np.where(probas_val[i, :] == maxInRows[i])[0][0])
        e = sorted(range(len(maxInRows)), key=maxInRows.__getitem__)
        selection = [e[-1 * toquery :]]
        return selection


class SSLselectionModified(BaseSelectionFunction):

    modified = True

    @staticmethod
    def select(probas_val, toquery):
        return SSLselection.select(probas_val, toquery)


class NOSSLselection(BaseSelectionFunction):
    pass


def get_k_random_samples(initial_labeled_samples, X_train_full, y_train_full):
    permutation = np.random.choice(
        y_train_full.shape[0], initial_labeled_samples, replace=False
    )
    print()
    print("Initial random chosen samples", permutation.shape),

    X_train = X_train_full.iloc[permutation].values
    y_train = y_train_full.iloc[permutation].values
    X_train = X_train.reshape((X_train.shape[0], -1))
    return (permutation, X_train, y_train)


def get_k_random_samples_stratified(
    initial_labeled_samples, X_train_full, y_train_full, LR
):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - LR, random_state=23)
    for tr, ts in sss.split(X_train_full, y_train_full):
        X_train = X_train_full.loc[tr]
        y_train = y_train_full.loc[tr]
    permutation = np.array(tr)
    print("Initial random chosen samples", permutation.shape),
    X_train = X_train_full.iloc[permutation].values
    y_train = y_train_full.iloc[permutation].values
    X_train = X_train.reshape((X_train.shape[0], -1))
    return (permutation, X_train, y_train)


class TheAlgorithm(object):

    accuracies = []
    precision, recall, fscore = [], [], []

    def __init__(
        self,
        initial_labeled_samples,
        model_object,
        selection_function,
        L0,
        selection_function_ssl,
        ssl_ratio,
        permutation,
        X_train,
        y_train,
        X_validate,
        y_validate,
    ):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function
        self.L0 = L0
        self.sample_selection_function_ssl = selection_function_ssl
        self.ssl_ratio = ssl_ratio
        self.permutation = permutation
        self.X_train = X_train
        self.y_train = y_train
        self.X_validate = X_validate
        self.y_validate = y_validate

    def run(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        permutation,
        X_labeled,
        y_labeled,
        X_validate,
        y_validate,
    ):

        self.queried = 0
        self.samplecount = [self.initial_labeled_samples]

        X_unlabeled = np.array([])
        y_unlabeled = np.array([])
        X_unlabeled = np.copy(X_train)
        y_unlabeled = np.copy(y_train)
        X_unlabeled = np.delete(X_unlabeled, permutation, axis=0)
        y_unlabeled = np.delete(y_unlabeled, permutation, axis=0)

        self.clf_model = TrainModel(self.model_object)

        (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
            X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
        )
        print("Initial labeled set:", X_labeled.shape)
        active_iteration = 0

        if not hasattr(self.sample_selection_function_ssl, "modified"):
            X_unlabeled = np.concatenate((X_unlabeled, X_validate))
            y_unlabeled = np.concatenate((y_unlabeled, y_validate))

        self.clf_model.get_test_accuracy(active_iteration, y_test, "Initial stage")

        combined = (
            self.sample_selection_function != NoActiveLearningSelection
            and self.sample_selection_function_ssl != NOSSLselection
        )
        active_learning = (
            self.sample_selection_function != NoActiveLearningSelection
            and self.sample_selection_function_ssl == NOSSLselection
        )
        ssl_learning = (
            self.sample_selection_function == NoActiveLearningSelection
            and self.sample_selection_function_ssl != NOSSLselection
        )
        irrelevant = (
            self.sample_selection_function == NoActiveLearningSelection
            and self.sample_selection_function_ssl == NOSSLselection
        )

        while self.queried < max_queried:
            active_iteration += 1

            if irrelevant:
                print("Final active learning accuracies")
                break
            if combined:
                # Get validation probabilities
                probas_val = self.clf_model.model_object.classifier.predict_proba(
                    X_unlabeled
                )

                # Select samples using a selection function
                uncertain_samples = self.sample_selection_function.select(
                    probas_val, self.initial_labeled_samples
                )

                # Get the uncertain samples from the validation set
                X_labeled = np.concatenate((X_labeled, X_unlabeled[uncertain_samples]))
                y_labeled = np.concatenate((y_labeled, y_unlabeled[uncertain_samples]))
                self.samplecount.append(X_train.shape[0])

                X_unlabeled = np.delete(X_unlabeled, uncertain_samples, axis=0)
                y_unlabeled = np.delete(y_unlabeled, uncertain_samples, axis=0)

                self.queried += self.initial_labeled_samples
                (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                    X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                )
                self.clf_model.get_test_accuracy(
                    active_iteration, y_test, "Active Learning"
                )

                # SSL
                probas_val_ssl = self.clf_model.model_object.classifier.predict_proba(
                    X_unlabeled
                )

                uncertain_samples_ssl = self.sample_selection_function_ssl.select(
                    probas_val_ssl, self.initial_labeled_samples * self.ssl_ratio
                )

                # Use of validation set
                if hasattr(self.sample_selection_function_ssl, "modified"):
                    initial = self.clf_model.model_object.classifier.fit(
                        X_labeled, y_labeled
                    )
                    aa = initial.score(X_validate, y_validate)

                    X_labeled_check = np.concatenate(
                        (X_labeled, X_unlabeled[uncertain_samples_ssl])
                    )
                    y_labeled_check = np.concatenate(
                        (y_labeled, y_unlabeled[uncertain_samples_ssl])
                    )

                    final = self.clf_model.model_object.classifier.fit(
                        X_labeled_check, y_labeled_check
                    )
                    bb = final.score(X_validate, y_validate)

                    if aa < bb:
                        X_labeled = np.concatenate(
                            (X_labeled, X_unlabeled[uncertain_samples_ssl])
                        )
                        y_labeled = np.concatenate(
                            (y_labeled, y_unlabeled[uncertain_samples_ssl])
                        )
                else:
                    X_labeled = np.concatenate(
                        (X_labeled, X_unlabeled[uncertain_samples_ssl])
                    )
                    y_labeled = np.concatenate(
                        (y_labeled, y_unlabeled[uncertain_samples_ssl])
                    )

                self.samplecount.append(X_labeled.shape[0])

                X_unlabeled = np.delete(X_unlabeled, uncertain_samples_ssl, axis=0)
                y_unlabeled = np.delete(y_unlabeled, uncertain_samples_ssl, axis=0)

                self.queried += self.initial_labeled_samples * self.ssl_ratio
                (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                    X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                )
                self.clf_model.get_test_accuracy(
                    active_iteration, y_test, "Semi-supervised Learning"
                )

            if ssl_learning:
                probas_val_ssl = self.clf_model.model_object.classifier.predict_proba(
                    X_unlabeled
                )

                uncertain_samples_ssl = self.sample_selection_function_ssl.select(
                    probas_val_ssl, self.initial_labeled_samples
                )

                # Use of validation set
                if hasattr(self.sample_selection_function_ssl, "modified"):
                    initial = self.clf_model.model_object.classifier.fit(
                        X_labeled, y_labeled
                    )
                    aa = initial.score(X_validate, y_validate)

                    X_labeled_check = np.concatenate(
                        (X_labeled, X_unlabeled[uncertain_samples_ssl])
                    )
                    y_labeled_check = np.concatenate(
                        (y_labeled, y_unlabeled[uncertain_samples_ssl])
                    )

                    final = self.clf_model.model_object.classifier.fit(
                        X_labeled_check, y_labeled_check
                    )
                    bb = final.score(X_validate, y_validate)

                    if aa < bb:
                        X_labeled = np.concatenate(
                            (X_labeled, X_unlabeled[uncertain_samples_ssl])
                        )
                        y_labeled = np.concatenate(
                            (y_labeled, y_unlabeled[uncertain_samples_ssl])
                        )
                else:
                    X_labeled = np.concatenate(
                        (X_labeled, X_unlabeled[uncertain_samples_ssl])
                    )
                    y_labeled = np.concatenate(
                        (y_labeled, y_unlabeled[uncertain_samples_ssl])
                    )

                self.samplecount.append(X_labeled.shape[0])

                X_unlabeled = np.delete(X_unlabeled, uncertain_samples_ssl, axis=0)
                y_unlabeled = np.delete(y_unlabeled, uncertain_samples_ssl, axis=0)

                self.queried += self.initial_labeled_samples
                (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                    X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                )
                self.clf_model.get_test_accuracy(
                    active_iteration, y_test, "Semi-supervised Learning"
                )

                if self.ssl_ratio > 1:

                    # SSL
                    classifier = self.clf_model.model_object.classifier
                    probas_val_ssl = classifier.predict_proba(X_unlabeled)

                    uncertain_samples_ssl = self.sample_selection_function_ssl.select(
                        probas_val_ssl, self.initial_labeled_samples * self.ssl_ratio
                    )

                    # Use of validation set
                    if hasattr(self.sample_selection_function_ssl, "modified"):
                        initial = self.clf_model.model_object.classifier.fit(
                            X_labeled, y_labeled
                        )
                        aa = initial.score(X_validate, y_validate)

                        X_labeled_check = np.concatenate(
                            (X_labeled, X_unlabeled[uncertain_samples_ssl])
                        )
                        y_labeled_check = np.concatenate(
                            (y_labeled, y_unlabeled[uncertain_samples_ssl])
                        )

                        final = self.clf_model.model_object.classifier.fit(
                            X_labeled_check, y_labeled_check
                        )
                        bb = final.score(X_validate, y_validate)

                        if aa < bb:
                            X_labeled = np.concatenate(
                                (X_labeled, X_unlabeled[uncertain_samples_ssl])
                            )
                            y_labeled = np.concatenate(
                                (y_labeled, y_unlabeled[uncertain_samples_ssl])
                            )
                    else:
                        X_labeled = np.concatenate(
                            (X_labeled, X_unlabeled[uncertain_samples_ssl])
                        )
                        y_labeled = np.concatenate(
                            (y_labeled, y_unlabeled[uncertain_samples_ssl])
                        )

                    self.samplecount.append(X_labeled.shape[0])

                    X_unlabeled = np.delete(X_unlabeled, uncertain_samples_ssl, axis=0)
                    y_unlabeled = np.delete(y_unlabeled, uncertain_samples_ssl, axis=0)

                    self.queried += self.initial_labeled_samples * self.ssl_ratio
                    (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                        X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                    )
                    self.clf_model.get_test_accuracy(
                        active_iteration, y_test, "Semi-supervised Learning"
                    )

            if active_learning:
                # Get validation probabilities
                probas_val = self.clf_model.model_object.classifier.predict_proba(
                    X_unlabeled
                )

                # Select samples using a selection function
                uncertain_samples = self.sample_selection_function.select(
                    probas_val, self.initial_labeled_samples
                )

                # Get the uncertain samples from the validation set
                X_labeled = np.concatenate((X_labeled, X_unlabeled[uncertain_samples]))
                y_labeled = np.concatenate((y_labeled, y_unlabeled[uncertain_samples]))
                self.samplecount.append(X_train.shape[0])

                X_unlabeled = np.delete(X_unlabeled, uncertain_samples, axis=0)
                y_unlabeled = np.delete(y_unlabeled, uncertain_samples, axis=0)

                self.queried += self.initial_labeled_samples
                (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                    X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                )
                self.clf_model.get_test_accuracy(
                    active_iteration, y_test, "Active Learning"
                )

                # Again AL
                if self.ssl_ratio > 1:
                    probas_val = self.clf_model.model_object.classifier.predict_proba(
                        X_unlabeled
                    )

                    # Select samples using a selection function
                    uncertain_samples = self.sample_selection_function.select(
                        probas_val, self.initial_labeled_samples * self.ssl_ratio
                    )

                    # Get the uncertain samples from the validation set
                    X_labeled = np.concatenate(
                        (X_labeled, X_unlabeled[uncertain_samples])
                    )
                    y_labeled = np.concatenate(
                        (y_labeled, y_unlabeled[uncertain_samples])
                    )
                    self.samplecount.append(X_train.shape[0])

                    X_unlabeled = np.delete(X_unlabeled, uncertain_samples, axis=0)
                    y_unlabeled = np.delete(y_unlabeled, uncertain_samples, axis=0)

                    self.queried += self.initial_labeled_samples * self.ssl_ratio
                    (X_labeled, X_unlabeled, X_test) = self.clf_model.train(
                        X_labeled, y_labeled, X_unlabeled, X_test, "balanced"
                    )
                    self.clf_model.get_test_accuracy(
                        active_iteration, y_test, "Active Learning"
                    )

        print("Final active learning accuracies", self.clf_model.accuracies)

        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_test, y_test


def experiment(
    d,
    models,
    selection_functions,
    Ks,
    repeats,
    contfrom,
    L0,
    selection_functions_ssl,
    ssl_ratio,
    X_train,
    y_train,
    X_test,
    y_test,
    permutation,
    X_labeled,
    y_labeled,
    X_validate,
    y_validate,
):

    print("Stopping at:", max_queried)
    count = 0
    for model_object in models:
        if model_object.__name__ not in d:
            d[model_object.__name__] = {}

        for selection_function in selection_functions:
            if selection_function.__name__ not in d[model_object.__name__]:
                d[model_object.__name__][selection_function.__name__] = {}

            for selection_function_ssl in selection_functions_ssl:
                if selection_function_ssl.__name__ not in d[model_object.__name__]:
                    d[model_object.__name__][selection_function.__name__][
                        selection_function_ssl.__name__
                    ] = {}

                for k in Ks:
                    d[model_object.__name__][selection_function.__name__][
                        selection_function_ssl.__name__
                    ][str(k)] = []

                    for i in range(0, repeats):
                        count += 1
                        print("--> Number of models", count)
                        X_train_exp = X_train[i]
                        y_train_exp = y_train[i]
                        X_test_exp = X_test[i]
                        y_test_exp = y_test[i]
                        X_validate_exp = X_validate[i]
                        y_validate_exp = y_validate[i]

                        if count >= contfrom:
                            print(
                                (
                                    "Count = %s, using model = %s, "
                                    "selection_function = %s, "
                                    "selection_function_ssl = %s, k = %s, "
                                    "iteration = %s., initial L0 set = %d."
                                )
                                % (
                                    count,
                                    model_object.__name__,
                                    selection_function.__name__,
                                    selection_function_ssl.__name__,
                                    k,
                                    i,
                                    L0,
                                )
                            )
                            alg = TheAlgorithm(
                                k,
                                model_object,
                                selection_function,
                                L0,
                                selection_function_ssl,
                                ssl_ratio,
                                permutation,
                                X_train,
                                y_train,
                                X_validate_exp,
                                y_validate_exp,
                            )

                            mytest = alg.run(
                                X_train_exp,
                                y_train_exp,
                                X_test_exp,
                                y_test_exp,
                                permutation,
                                X_labeled,
                                y_labeled,
                                X_validate_exp,
                                y_validate_exp,
                            )
                            d[model_object.__name__][selection_function.__name__][
                                selection_function_ssl.__name__
                            ][str(k)].append(
                                [
                                    alg.clf_model.accuracies,
                                    alg.clf_model.precision,
                                    alg.clf_model.recall,
                                    alg.clf_model.fscore,
                                ]
                            )
                            print(MSG_END)
    return d, mytest


choice = "CHAINS"

for LR in [0.02]:
    ssl_ration = [1, 3]
    query_pools = [200, 200]

    if choice == "voice":
        (X, y) = download_dataset(1)
    elif choice == "ANAD":
        (X, y) = download_dataset(2)
    elif choice == "CHAINS":
        (X, y) = download_dataset(3)
    else:
        exit()

    dataset_size = X.shape[0]
    testset_size = 0.1 * dataset_size
    validateset_size = testset_size
    repeats = 3

    # Split to validate set
    (X_train_full, y_train_full, X_test, y_test) = split_alssl(
        X, y, repeats, testset_size / dataset_size
    )
    (X_train, y_train, X_validate, y_validate, a, b) = split_alssl_lists(
        X_train_full, y_train_full, 1, validateset_size / X_train_full[0].shape[0]
    )

    # Remove class from X
    X_train_full = remove_class(X_train_full)
    X_train = remove_class(X_train)
    X_test = remove_class(X_test)
    X_validate = remove_class(X_validate)

    classes = len(np.unique(y))
    print("L + U    :", X_train[0].shape, y_train[0].shape)
    print("Validate :", X_validate[0].shape, y_validate[0].shape)
    print("test     :", X_test[0].shape, y_test[0].shape)

    print("unique classes", classes)
    trainset_size = X_train[0].shape[0]
    L0 = int(trainset_size * LR)

    # Apply for each repeat (seed)
    for pointer in range(0, len(ssl_ration)):
        print(MSG_START.format(ssl_ration[pointer], LR))

        (permutation, X_labeled, y_labeled) = get_k_random_samples_stratified(
            L0, X_train[pointer], y_train[pointer], LR
        )  # Selects the initial L set
        ssl_ratio = ssl_ration[pointer]
        max_queried = query_pools[pointer]

        d = {}
        stopped_at = -1

        Ks_str = ["2", "5", "10", "25"]
        Ks = [2, 5, 10, 25]

        selection_functions = [
            MarginSamplingSelection,
            EntropySelection,
            MinStdSelection,
            RandomSelection,
            NoActiveLearningSelection,
        ]
        selection_functions_str = [
            "MarginSamplingSelection",
            "EntropySelection",
            "MinStdSelection",
            "RandomSelection",
            "NoActiveLearningSelection",
        ]

        selection_functions_ssl = [SSLselection, SSLselectionModified, NOSSLselection]
        selection_functions_ssl_str = [
            "SSLselection",
            "SSLselectionModified",
            "NOSSLselection",
        ]

        models = [
            KNNModel,
            RfModel,
            NBModel,
            ExtModel,
            MLPModel,
            MLPModel_2layers,
            MLPModel_3layers,
        ]
        models_str = [
            "KNNModel",
            "RfModel",
            "NBModel",
            "ExtModel",
            "MLPModel",
            "MLPModel_2layers",
            "MLPModel_3layers",
        ]

        # X_train is L + U
        print(models)
        d, mytest = experiment(
            d,
            models,
            selection_functions,
            Ks,
            repeats,
            stopped_at + 1,
            L0,
            selection_functions_ssl,
            ssl_ratio,
            X_train,
            y_train,
            X_test,
            y_test,
            permutation,
            X_labeled,
            y_labeled,
            X_validate,
            y_validate,
        )

        for i in mytest:
            print(i.shape)

        with open(
            "stam2_ncaa_"
            + choice
            + "_LR_"
            + str(LR)
            + "_ratio_1_"
            + str(ssl_ratio)
            + "_"
            + str(max_queried)
            + "_insta.pickle",
            "wb",
        ) as f:
            pickle.dump(d, f, protocol=2)
        with open(
            "stam3_ncaa_"
            + choice
            + "_LR_"
            + str(LR)
            + "_ratio_1_"
            + str(ssl_ratio)
            + "_"
            + str(max_queried)
            + "_insta.pickle",
            "wb",
        ) as f:
            pickle.dump(d, f, protocol=3)
        del d, mytest
