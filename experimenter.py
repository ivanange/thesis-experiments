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
import describe_test
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearnex import patch_sklearn
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
def train(file, kernel='power', cv=False, alpha=0.05, gamma=0.5, degree=3, beta=0.8, random_state=0):
    '''Train a model to predict survival from the given data.'''

    # set random state
    np.random.seed(random_state)
    # np.random.set_state(random_state)
    kernels = {
        'power': lambda x: gramMatrix(x, x, lambda x1, x2: np.clip(-np.linalg.norm(x1 - x2)**beta, 2e-100, 2e100)),
        'log': lambda x:  gramMatrix(x, x, lambda x1, x2: np.clip(-np.log(1 + np.linalg.norm(x1 - x2, axis=0)**beta), 2e-100, 2e100)),
        'mixture': lambda x: np.clip(alpha * pairwise_kernels(x, metric='rbf', gamma=gamma) + (1-alpha)*pairwise_kernels(x, metric='poly', degree=degree), 2e-100, 2e100),
    }

    # read data from npy file
    filename = file.endswith(
        '.npy') and file or f'{file}.npy'
    dataset = np.load(os.path.join('features', filename), allow_pickle=True)

    # shuffle dataset
    np.random.shuffle(dataset)

    # extract features and labels
    y = list()
    for indicator, time in dataset[:, [-1, -2]]:
        y.append((indicator, time))
    y = np.array(y, dtype=[('status', bool), ('time', '<f8')]).flatten()

    # y = dataset[:, [-1, -2]]
    print(y)
    x = np.delete(dataset, [-3, -2, -1], axis=1).astype(np.float16)
    # x = np.array(x)

    # evaluate kernel matrix
    kernel_name = kernel
    gamma = 1/x.shape[1]
    kernel = kernels[kernel_name]
    kernel_matrix = kernel(x)
    # kernel_matrix = StandardScaler().fit(kernel_matrix).transform(kernel_matrix)

    kernel_matrix

    # setup model
    model = FastKernelSurvivalSVM(
        kernel='precomputed', random_state=random_state, timeit=1, alpha=1, rank_ratio=1, max_iter=100)

    if not cv:
        model.fit(kernel_matrix, y)
    else:
        # train model with CV
        cv_results = cross_validate(model, kernel_matrix, y, cv=4, scoring=make_scorer(
            custom_scoring_function, greater_is_better=True), return_train_score=True, return_estimator=True)
        score = np.mean(cv_results['test_score'])

        # get best estimator
        model = cv_results['estimator'][np.argmax(cv_results['test_score'])]
        model.cv_results = cv_results
        model.score = score
        print(cv_results)

    # score = score_survival_model(model, kernel_matrix, y)
    # print(f'Concordance index: {score}')
    print(model.coef_)

    # save the model to disk
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    filename = os.path.join('models', f'{file}_{kernel_name}_{now}_model.sav')
    dump(model, open(filename, 'wb'))

    # %%
    # load the model from disk
    loaded_model = load(open(filename, 'rb'))
    print(loaded_model.coef_)

    # %%
    # score = score_survival_model(loaded_model, kernel_matrix, y)
    # score


if __name__ == '__main__':
    main()
