# %%
from mando import command, main
import os
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import cross_validate
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.metrics import concordance_index_censored
import pickle
from datetime import datetime
import describe
import kernel_matrices
import analysis
import describe_test
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearnex import patch_sklearn
from sklearn.model_selection import KFold
patch_sklearn(global_patch=True)


def gramMatrix(X1, X2, K_function):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2)
    return gram_matrix


def score_survival_model(model, X, y):
    prediction = model.predict(X)
    result = concordance_index_censored(
        y['status'], y['time'], prediction)
    return result[0]


def custom_scoring_function(y, prediction):
    result = concordance_index_censored(
        y['status'], y['time'], prediction)
    return result[0]


@command
def train(feature, kernel, cv=4, rank_ratio=1, random_state=42):
    '''Train a model to predict survival from the given data.'''

    # set random state
    np.random.seed(random_state)
    kernel_name = kernel

    kernel_matrix = np.load(
        f'kernel_matrices/{feature}_{kernel_name}_kernel_matrix.npy')
    labels = np.load(f'kernel_matrices/{feature}_{kernel_name}_labels.npy')

    # extract features and labels
    y = list()
    for indicator, time in labels[:, [-1, -2]]:
        y.append((indicator, time))
    y = np.array(y, dtype=[('status', bool), ('time', '<f8')]).flatten()

    # setup model
    model = FastKernelSurvivalSVM(
        kernel='precomputed', rank_ratio=rank_ratio, random_state=random_state, timeit=1, alpha=1, max_iter=100)

    if not cv:
        model.fit(kernel_matrix, y)
    else:
        # train model with CV
        kf = KFold(cv, random_state=random_state, shuffle=True)
        cv_results = cross_validate(model, kernel_matrix, y, cv=kf, scoring=make_scorer(
            custom_scoring_function, greater_is_better=True), return_train_score=True, return_estimator=True)
        score = np.mean(cv_results['test_score'])

        # get best estimator
        model = cv_results['estimator'][np.argmax(cv_results['test_score'])]
        model.cv_results = cv_results
        # model.score = [test_index for train_index, test_index in kf.split(X)]

    model.feature = feature
    model.precomputed_kernel = kernel_name

    # save the model to disk
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    filename = os.path.join(
        'models', f'{feature}_{kernel_name}_ratio_{rank_ratio}_{now}_model.sav')
    dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    main()
