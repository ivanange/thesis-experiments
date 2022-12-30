# %%
from mando import command, main
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_kernels
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.column import encode_categorical
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.metrics import concordance_index_censored
import pickle
from datetime import datetime


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


@command
def train(file, kernel='power', alpha=0.05, gamma=0.5, degree=2, beta=2, random_state=0):
    '''Train a model to predict survival from the given data.'''

    # set random state
    random_state = random_state

    kernels = {
        'power': lambda x: gramMatrix(x, x, lambda x1, x2: -np.linalg.norm(x1 - x2)**beta),
        'log': lambda x:  gramMatrix(x, x, lambda x1, x2: -np.log(1 + np.linalg.norm(x1 - x2, axis=0)**beta)),
        'mixture': lambda x: alpha * pairwise_kernels(x, metric='rbf', gamma=gamma) + (1-alpha)*pairwise_kernels(x, metric='poly', degree=degree),
    }

    # read data from file
    filename = file.endswith(
        '.csv') and file or f'{file}.csv'
    dataset = pd.read_csv(os.path.join('features', filename))

    # shuffle dataset
    dataset = dataset.sample(frac=1, random_state=random_state)

    # extract features and labels
    y = list()
    for index, row in dataset.iloc[:, [-1, -2]].iterrows():
        y.append((row[-2], row[-1]))
    y = np.array(y, dtype=[('status', bool), ('time', '<f8')])
    # print(y)
    x = dataset.drop(columns=dataset.iloc[:, [-3, -2, -1]])
    x = np.array(x)

    # evaluate kernel matrix
    kernel_name = kernel
    kernel = kernels[kernel_name]
    kernel_matrix = kernel(x)
    kernel_matrix

    # setup model
    model = FastKernelSurvivalSVM(
        kernel='precomputed', random_state=random_state)

    # train model
    model.fit(kernel_matrix, y)

    score = score_survival_model(model, kernel_matrix, y)
    print(f'Concordance index: {score}')

    # save the model to disk
    now = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    filename = os.path.join('models', f'{kernel_name}_{now}_model.sav')
    pickle.dump(model, open(filename, 'wb'))

    # %%
    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))

    # %%
    # score = score_survival_model(loaded_model, kernel_matrix, y)
    # score


if __name__ == '__main__':
    main()
