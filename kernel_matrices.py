import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearnex import patch_sklearn
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from mando import command, main

patch_sklearn(global_patch=True)

experiments = {
    'sift': {
        'dataset':  'features/dataset_reduced.csv_SIFT_.npy',
        'model': {
            'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
            'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
            'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
        },
        'scores': {
            'log': 'scores/dataset_reduced.csv_SIFT_.npy_log_26-01-2023 06-55-40_model.sav',
            'power': 'scores/dataset_reduced.csv_SIFT_.npy_power_25-01-2023 16-32-18_model.sav',
            'mixture': 'scores/dataset_reduced.csv_SIFT_.npy_mixture_25-01-2023 22-27-27_model.sav',
        },
    },
    'hog': {
        'dataset':  'features/dataset_reduced.csv_HOG_.npy',
        'model': {
            'log': 'models/dataset_reduced.csv_HOG_.npy_log_01-02-2023 06-57-32_model.sav',
            'power': 'models/dataset_reduced.csv_HOG_.npy_power_01-02-2023 07-46-17_model.sav',
            'mixture': 'models/dataset_reduced.csv_HOG_.npy_mixture_03-02-2023 13-19-27_model.sav',
        },
        'scores': {
            'log': 'scores/dataset_reduced.csv_HOG_.npy_log_20-01-2023 22-29-15_model.sav',
            'power': 'scores/dataset_reduced.csv_HOG__power_19-01-2023 18-42-58_model.sav',
            'mixture': 'scores/dataset_reduced.csv_HOG_.npy_mixture_23-01-2023 14-34-38_model.sav',
        },
    },
    'kdesa': {
        'dataset':  'features/dataset_reduced.csv_KDESA_.npy',
        'model': {
            'log': 'models/dataset_reduced.csv_KDESA_.npy_log_03-02-2023 22-47-11_model.sav',
            'power': 'models/dataset_reduced.csv_KDESA_.npy_power_03-02-2023 22-44-27_model.sav',
            'mixture': 'models/dataset_reduced.csv_KDESA_.npy_mixture_03-02-2023 13-01-11_model.sav',
        },
        'scores': {
            'log': 'scores/dataset_reduced.csv_KDESA_.npy_log_21-01-2023 07-35-41_model.sav',
            'power': 'scores/dataset_reduced.csv_KDESA_.npy_power_21-01-2023 06-10-39_model.sav',
            'mixture': 'scores/dataset_reduced.csv_KDESA_.npy_mixture_23-01-2023 14-41-13_model.sav',
        },
    },
}

alpha = 0.05
gamma = 0.5
degree = 3
beta = 0.8
random_state = 0


def gramMatrix(X1, X2, K_function):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2)
    return gram_matrix


kernels = {
    'power': lambda x, y: gramMatrix(x, y, lambda x1, x2: np.clip(-np.linalg.norm(x1 - x2)**beta, 2e-100, 2e100)),
    'log': lambda x, y:  gramMatrix(x, y, lambda x1, x2: np.clip(-np.log(1 + np.linalg.norm(x1 - x2, axis=0)**beta), 2e-100, 2e100)),
    'mixture': lambda x, y: np.clip(alpha * pairwise_kernels(x, y, metric='rbf', gamma=gamma) + (1-alpha)*pairwise_kernels(x, y, metric='poly', degree=degree), 2e-100, 2e100),
}


def load_dataset(feature):
    data = experiments[feature]
    dataset = np.load(data['dataset'])
    x = np.delete(dataset, [-3, -2, -1], axis=1)
    # labels: 1 - invalid, 0 - valid
    labels = dataset[:, [-3, -2, -1]]
    return x, labels


def predict(feature, kernel, model, x):
    x_train, _ = load_dataset(feature)
    kernel_matrix = kernels[kernel](x, x_train.T)
    return model.predict(kernel_matrix)


@command
def kernel_matrices(feature):
    for kernel_name in ['log', 'power', 'mixture']:
        kernel_matrix(feature, kernel_name)


@command
def kernel_matrix(feature, kernel_name):
    x, labels = load_dataset(feature)
    kernel = kernels[kernel_name]
    kernel_matrix = kernel(x, x)
    # save kernel matrix with labels
    np.save(
        f'kernel_matrices/{feature}_{kernel_name}_kernel_matrix.npy', kernel_matrix)
    np.save(f'kernel_matrices/{feature}_{kernel_name}_labels.npy', labels)
