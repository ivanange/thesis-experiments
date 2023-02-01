import glob
from mando import command, main
import os
import numpy as np
import pandas as pd
from skimage import transform
from load_features import get_feature_extractor
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, hog, SIFT
from kernel_descriptors_extractor import KernelDescriptorsExtractor
from skimage import exposure
from skimage.transform import rescale, resize, downscale_local_mean
import pickle
import traceback

# defining constants
CORRECTION_FACTOR = 2
TARGET_SIZE = (256, 256)
FEATURE_TARGET_SIZE = 100000
DATASET_DIR = 'dataset'
FEATURES_DIR = 'features'
MAX_FEATURE_SIZE = {'KDESA': 400000, 'HOG': 328888, 'SIFT': 328888}
CATEGORY_TO_INT = {'invalid': 0, 'underwear': 1, 'valid': 2}

# define image transforms
# create transformations with different values of scale: 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, rotation: 30, 60, 90, 150, 180, 270, translation: (10,0), (20, 40), (40, 80), (80, 160), (-10, 10), (-20, -40), (-40, -80), (-80, -160), shear: 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5
transform_combinations = [
    {},
    # {'scale': 0.25},
    # {'scale': 0.5},
    # {'scale': 2},
    # {'scale': 2.5},
    {'rotation': 60},
    # {'rotation': 90},
    {'rotation': 150},
    {'rotation': 300},
    # {'scale': 0.25, 'rotation': 90},
    # {'scale': 0.25, 'rotation': 150},
    # {'scale': 0.25, 'rotation': 180},
    # {'scale': 2, 'rotation': 90},
    # {'scale': 2, 'rotation': 150},
    # {'scale': 2, 'rotation': 180},
]


def transforms_string(transform_combination): return '-'.join(
    map(lambda item: f'{item[0]}_{item[1]}', transform_combination.items()))


def get_descriptor_extractor(descriptor_name):
    '''get descriptor extractor function'''
    if descriptor_name == 'KDESA':
        KDESA_extractor_file = 'KDESA_extractor.pkl'
        if not os.path.exists(KDESA_extractor_file):
            KDESAExtractor = KernelDescriptorsExtractor()
            pickle.dump(KDESAExtractor, open(KDESA_extractor_file, 'wb'))
        else:
            KDESAExtractor = pickle.load(open(KDESA_extractor_file, 'rb'))

        def descriptor(image): return KDESAExtractor.predict(
            np.array([image]), match_kernel='all')[0]

    elif descriptor_name == 'SIFT':
        SIFTExtractor = SIFT()

        def descriptor(image): return ([SIFTExtractor.detect_and_extract(
            image), np.array(SIFTExtractor.descriptors).flatten()])[1]

    elif descriptor_name == 'HOG':
        def descriptor(image): return hog(
            image, feature_vector=True, channel_axis=2)

    else:
        Exception(f'Unknown descriptor {descriptor_name}')
        # traceback.print_exc()

    return descriptor


def extract_features_from_image(filename, descriptor, transform_combination, sift=False):
    try:

        # load image as numpy array
        image = io.imread(filename)
        print(f"Loaded image of size {image.shape}")

        # rescale instead of resizing to avoid distortion
        max_size = max(image.shape)
        target_scale = min(TARGET_SIZE) / max_size
        print(
            f"Rescaling image by {target_scale} to fit target size {TARGET_SIZE}")
        image = rescale(image, target_scale,
                        anti_aliasing=True, multichannel=True)

        # apply image transform combination
        print(
            f"Applying transform combination {transforms_string(transform_combination)}")
        image = transform.warp(image, transform.AffineTransform(
            **transform_combination).inverse)

        # convert image to grayscale if using SIFT descriptor
        if sift:
            print("Converting image to grayscale for SIFT descriptor")
            image = rgb2gray(image)

        # adjust gamma to improve contrast
        print(f"Adjusting gamma by {CORRECTION_FACTOR} to improve contrast")
        image = exposure.adjust_gamma(image, CORRECTION_FACTOR)

        # extract features
        print("Extracting features")
        feature = descriptor(image)
        print(f"Extracted feature of size {feature.shape}")

        return feature

    except Exception as e:
        print(f"Error extracting features for {filename}")
        print(e)
        traceback.print_exc()
        return np.zeros(1)


def extract_features_from_images_dataset(dataset_file, descriptor_name, descriptor, transform_combination):

    # create empty features array of size (n_images)
    features = list()
    max_feature_size = MAX_FEATURE_SIZE[descriptor_name]

    # load images dataset csv file
    dataset = pd.read_csv(dataset_file)

    # for each image: apply transform combination, extract features
    for index, row in dataset.iterrows():
        filename = os.path.join(DATASET_DIR, row['file_path'])
        print(f"Processing {index + 1} - {filename}")
        feature = extract_features_from_image(
            filename, descriptor, transform_combination, descriptor_name == 'SIFT')

        # pad each row in features to max_feature_size
        print(f"Padding features to max size {max_feature_size}")
        features.append(np.pad(feature, pad_width=(
            0, max_feature_size - len(feature)), mode='constant', constant_values=0))

    # convert features to numpy array
    features = np.asarray(features, dtype=np.float16)
    print(f"Features shape: {features.shape}")

    # print(f"Features shape before: {features.shape}")
    print(f"adding category, time and censored columns")
    features = np.concatenate((
        features,
        np.asarray(list(map(
            lambda category: CATEGORY_TO_INT[category.split("/")[0]],
            dataset['file_path']
        )), dtype=np.uint).reshape(-1, 1),
        np.asarray(dataset['time'], dtype=np.uint).reshape(-1, 1),
        np.asarray(dataset['censored'] == 1,
                   dtype=np.bool_).reshape(-1, 1),
    ), axis=1, dtype=np.float16)
    # print(f"Features shape after: {features.shape}")
    # print(np.shape(features))

    filename = f"{dataset_file.split(os.sep)[-1]}_{descriptor_name}_{transforms_string(transform_combination)}.npy"
    print(f"Saving features to {filename}")

    # save dataset to file as {transform_combination}_{descriptor_name}.npy
    np.save(os.path.join(FEATURES_DIR,  filename), features)

    return features, dataset


def extract_features_from_dataset(dataset_file, descriptor, descriptor_name):
    '''extract features from dataset'''

    try:
        full_features = None

        # for each image transform combination
        for transform_combination in transform_combinations:
            print(
                f'Processing transform combination {transforms_string(transform_combination)}')
            (features, dataset) = extract_features_from_images_dataset(
                dataset_file, descriptor_name, descriptor, transform_combination)
            full_features = features if full_features is None else np.concatenate(
                (full_features, features), axis=0)

        # # convert full_features to numpy array
        full_features = np.asarray(full_features, dtype=np.float16)

        # # create full_features dataset
        n_transform_combinations = len(transform_combinations)
        full_features = np.concatenate((
            full_features,
            np.asarray(list(map(
                lambda category: CATEGORY_TO_INT[category.split("/")[0]],
                dataset['file_path']
            )) * n_transform_combinations, dtype=np.uint).reshape(-1, 1),
            np.asarray(
                list(dataset['time']) * n_transform_combinations, dtype=np.uint).reshape(-1, 1),
            np.asarray(list((dataset['censored']) == 1) * n_transform_combinations,
                       dtype=np.bool_).reshape(-1, 1),
        ), axis=1, dtype=np.float16)

        filename = f"{dataset_file.split(os.sep)[-1]}_{descriptor_name}_augmented.npy"
        print(filename)
        print(full_features.shape)
        # save dataset to file as {descriptor_name}_augmented.npy
        np.save(os.path.join(FEATURES_DIR,  filename), full_features)

    except Exception as e:
        print(
            f"Error extracting features from {dataset_file} using {descriptor_name} with {transform_combination}")
        print(e)
        traceback.print_exc()


@command
def describe(descriptor, datasets, random_seed=0):
    '''extract features using the precised descriptor'''

    # set random state
    np.random.seed(random_seed)

    # get descriptor extractor function
    descriptor_name = descriptor.upper()
    descriptor = get_descriptor_extractor(descriptor_name)
    print(f'Using {descriptor_name} descriptor')

    for dataset_file in glob.glob(datasets, recursive=True, root_dir=DATASET_DIR):
        dataset_file = os.path.join(DATASET_DIR, dataset_file)
        print(f"Processing dataset {dataset_file}")
        extract_features_from_dataset(
            dataset_file, descriptor, descriptor_name)
