# from PIL import Image
# from kernel_descriptors_extractor import KernelDescriptorsExtractor
# from sift_feature_extractor import SIFTFeatureExtractor
# from skimage.feature import match_descriptors, plot_matches, SIFT
# from skimage.color import rgb2gray
# from skimage import io
import numpy as np
from scipy.ndimage import generic_filter
from functools import reduce
from describe_test import describe_test
from describe import describe
from experimenter import train
from kernel_matrices import kernel_matrices, load_dataset, kernels
import pickle
import os

# describe('sift', 'dataset_reduced.csv')
# describe('kdesa', 'test-dataset.csv')
# describe('hog', 'test-dataset copy.csv')
# train('test-dataset copy.csv_HOG_augmented.npy', 'mixture')
# train('test-dataset copy.csv_HOG_augmented.npy', 'log', True)


def kernel_matrix(feature, kernel_name):
    x, labels = load_dataset(feature)
    sample_indices = np.random.choice(x.shape[0], 50, replace=False)
    sample = x[sample_indices]
    kernel = kernels[kernel_name]
    kernel_matrix = kernel(sample, sample)
    # save kernel matrix with labels
    print(kernel_matrix)


kernel_matrix('sift', 'power')
# feature = 'sift'
# kernel_name = 'power'
# kernel_matrix = np.load(
#     f'kernel_matrices/{feature}_{kernel_name}_kernel_matrix.npy')
# labels = np.load(f'kernel_matrices/{feature}_{kernel_name}_labels.npy')
# print(kernel_matrix)
# train('HOG_.npy')

# model = pickle.load(open(os.path.join(
#     'models', 'test-dataset copy.csv_HOG_augmented.npy_power_19-01-2023 12-17-16_model.sav'), 'rb'))
# print(model.score)
# print(model.cv_results)


# I = numpy.random.randint(30, size=(10, 10, 3))
# print(I.shape)


# def extractor(image):
#     global H, W, pos, result
#     H, W, _ = I.shape
#     result = numpy.empty((H, W, 3**2))

#     def binary_pattern_extractor_reduce(rest, c):
#         # global result, H, W, pos
#         global pos
#         pos = 0
#         generic_filter(I[:, :, c], binary_pattern_extractor,
#                        size=3, mode='constant', cval=0)
#         return numpy.concatenate((rest, result), axis=2) if type(rest) == numpy.ndarray else result

#     def binary_pattern_extractor(local_mat):
#         global result, H, W, pos
#         # local_mat[(len(local_mat)//2)+1]
#         print(local_mat, H, W, pos, pos//H, pos % H)
#         result[pos//H, pos %
#                H] = numpy.where(local_mat >= numpy.nanmedian(local_mat), 1, 0)
#         pos += 1
#         return 1

#     b = reduce(binary_pattern_extractor_reduce, [0, 1, 2], None)

#     print(b.shape)
#     print(b[1, 1, :])


# extractor(I)
