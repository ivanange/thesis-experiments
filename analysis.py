#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score
from kernel_matrices import predict, load_dataset
from mando import command

# In[7]:

experiment_results = {
    'sift': {
        'dataset':  'features/dataset_reduced.csv_SIFT_.npy',
        'model': {
            'rank': {
                'log': 'models/sift_log_ratio_1_01-03-2023 10-37-43_model.sav',
                'power': 'models/sift_power_ratio_1_01-03-2023 10-43-17_model.sav',
                'mixture': 'models/sift_mixture_ratio_1_01-03-2023 10-42-35_model.sav',
            },
            'reg': {
                'log': 'models/sift_log_ratio_0_01-03-2023 10-43-51_model.sav',
                'power': 'models/sift_power_ratio_0_01-03-2023 10-49-51_model.sav',
                'mixture': 'models/sift_mixture_ratio_0_01-03-2023 10-48-35_model.sav',
            }
        },
        'scores': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'times': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'thresholds': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'predictions': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
    },
    'hog': {
        'dataset':  'features/dataset_reduced.csv_HOG_.npy',
        'model': {
            'rank': {
                'log': 'models/hog_log_ratio_1_01-03-2023 10-51-06_model.sav',
                'power': 'models/hog_power_ratio_1_01-03-2023 11-02-55_model.sav',
                'mixture': 'models/hog_mixture_ratio_1_01-03-2023 10-51-38_model.sav',
            },
            'reg': {
                'log': 'models/hog_log_ratio_0_01-03-2023 11-00-55_model.sav',
                'power': 'models/hog_power_ratio_0_01-03-2023 11-01-27_model.sav',
                'mixture': 'models/hog_mixture_ratio_0_01-03-2023 11-01-11_model.sav',
            }
        },
        'scores': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'times': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'thresholds': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'predictions': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
    },
    'kdesa': {
        'dataset':  'features/dataset_reduced.csv_KDESA_.npy',
        'model': {
            'rank': {
                'log': 'models/kdesa_log_ratio_1_01-03-2023 10-12-46_model.sav',
                'power': 'models/kdesa_power_ratio_1_01-03-2023 10-35-16_model.sav',
                'mixture': 'models/kdesa_mixture_ratio_1_01-03-2023 10-36-10_model.sav',
            },
            'reg': {
                'log': 'models/kdesa_log_ratio_0_01-03-2023 10-18-32_model.sav',
                'power': 'models/kdesa_power_ratio_0_01-03-2023 10-35-34_model.sav',
                'mixture': 'models/kdesa_mixture_ratio_0_01-03-2023 10-23-47_model.sav',
            }
        },
        'scores': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'times': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'thresholds': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
        'predictions': {
            'rank': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            },
            'reg': {
                'log': 'models/dataset_reduced.csv_SIFT_.npy_log_01-02-2023 06-29-21_model.sav',
                'power': 'models/dataset_reduced.csv_SIFT_.npy_power_01-02-2023 07-07-27_model.sav',
                'mixture': 'models/dataset_reduced.csv_SIFT_.npy_mixture_31-01-2023 13-20-17_model.sav',
            }
        },
    },
}

# In[7]:


@command
def analyze(feature, size=0.2, random_state=42):
    data = experiment_results[feature]
    x, labels = load_dataset(feature)
    np.random.seed(random_state)
    # np.random.set_state(random_state)
    sample_indices = np.random.choice(
        x.shape[0], int(size * x.shape[0]), False)
    X, labels_sample = (x[sample_indices, :], labels[sample_indices, :])

    for (kernel_name, model_path) in data['model']['rank'].items():
        print(f'Kernel: {kernel_name}')
        model = load(open(model_path, 'rb'))
        print(model.kernel)
        model.precomputed_kernel = kernel_name
        model.kernel = "precomputed"
        dump(model, open(model_path, 'wb'))
        experiment_results[feature]['times']['rank'][kernel_name] = times = predict(
            feature, kernel_name, model, X)

        threshold = 0
        threshold_score = 0
        y_true = labels_sample[:, -1]
        for time in np.unique(times):
            prediction = times > (time - 1e-100)
            score = accuracy_score(y_true, prediction)
            (threshold, threshold_score) = (
                time, score) if score > threshold_score else (threshold, threshold_score)

        experiment_results[feature]['scores']['rank'][kernel_name] = threshold_score
        experiment_results[feature]['thresholds']['rank'][kernel_name] = threshold
        prediction = times > threshold
        experiment_results[feature]['predictions']['rank'][kernel_name] = prediction

    for (kernel_name, model_path) in data['model']['reg'].items():
        print(f'Kernel: {kernel_name}')
        model = load(open(model_path, 'rb'))
        print(model.kernel)
        model.precomputed_kernel = kernel_name
        model.kernel = "precomputed"
        experiment_results[feature]['times']['reg'][kernel_name] = times = predict(
            feature, kernel_name, model, X)

        threshold = 0
        threshold_score = 0
        print(labels[:, -1])
        y_true = labels_sample[:, -1]
        for time in np.unique(times):
            prediction = times > (time - 1e-100)
            score = accuracy_score(y_true, prediction)
            (threshold, threshold_score) = (
                time, score) if score > threshold_score else (threshold, threshold_score)

        experiment_results[feature]['scores']['reg'][kernel_name] = threshold_score
        experiment_results[feature]['thresholds']['reg'][kernel_name] = threshold
        prediction = times > threshold
        experiment_results[feature]['predictions']['reg'][kernel_name] = prediction

    # save results
    data = experiment_results[feature]
    dump(data, open(f'results/{feature}.pkl', 'wb'))
