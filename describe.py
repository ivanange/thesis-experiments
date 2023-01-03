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

# load images dataset csv file
dataset = pd.read_csv('./dataset/test-dataset.csv')

# set random state
np.random.seed(0)

# define image transforms
# create transformations with different values of scale: 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, rotation: 30, 60, 90, 150, 180, 270, translation: (10,0), (20, 40), (40, 80), (80, 160), (-10, 10), (-20, -40), (-40, -80), (-80, -160), shear: 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5
transform_combinations = [
    {},
    # {'scale': 0.25},
    # {'scale': 0.5},
    # {'scale': 2},
    # {'scale': 2.5},
    # {'rotation': 60},
    # {'rotation': 90},
    # {'rotation': 150},
    # {'rotation': 180},
    # {'scale': 0.25, 'rotation': 90},
    # {'scale': 0.25, 'rotation': 150},
    # {'scale': 0.25, 'rotation': 180},
    # {'scale': 2, 'rotation': 90},
    # {'scale': 2, 'rotation': 150},
    # {'scale': 2, 'rotation': 180},

]

# descriptor_names = {
#     'SIFT': lambda image: ([SIFTExtractor.detect_and_extract(image), np.array(SIFTExtractor.descriptors).flatten()])[1],
#     'HOG': lambda image: hog(image, feature_vector=True, channel_axis=2),
#     'KDESA': lambda image: KDESAExtractor.predict(np.array([image]))[0]
# }


@command
def describe(descriptor):
    '''extract features using the precised descriptor'''
    descriptor_name = descriptor.upper()
    if descriptor_name == 'KDESA':
        KDESAExtractor = KernelDescriptorsExtractor()

        def descriptor(image): return KDESAExtractor.predict(
            np.array([image]))[0]

    elif descriptor_name == 'SIFT':
        SIFTExtractor = SIFT()

        def descriptor(image): return ([SIFTExtractor.detect_and_extract(
            image), np.array(SIFTExtractor.descriptors).flatten()])[1]

    elif descriptor_name == 'HOG':
        def descriptor(image): return hog(
            image, feature_vector=True, channel_axis=2)

    else:
        Exception(f'Unknown descriptor {descriptor_name}')

    print(f'Using {descriptor_name} descriptor')

    max_full_feature_size = 0
    full_features = list()

    try:
        # for each image transform combination
        for transform_combination in transform_combinations:

            # create empty features array of size (n_images)
            features = list()
            max_feature_size = 0

            # for each image: apply transform combination, extract features
            for index, row in dataset.iterrows():
                try:
                    # apply image transform combination
                    image = io.imread(os.path.join(
                        'dataset', row['file_path']))
                    image = transform.warp(image, transform.AffineTransform(
                        **transform_combination).inverse)

                    # convert image to grayscale if not using kernel descriptors
                    if descriptor_name == 'SIFT':
                        image = rgb2gray(image)

                    image = exposure.adjust_gamma(image, 2)

                    # extract features, remember max_feature_size
                    feature = descriptor(image)
                    # print(feature.shape)
                    # print(feature)
                    features.append(feature)
                    max_feature_size = max(max_feature_size, len(feature))
                except Exception as e:
                    features.append(np.zeros(max_feature_size))
                    error = True
                    print(f"Error extracting features for {row['file_path']}")
                    print(e)

            # pad each row in features to max_feature_size
            for (i, feature) in enumerate(features):
                features[i] = np.pad(feature, pad_width=(
                    0, max_feature_size - len(feature)), mode='constant', constant_values=0)

            # convert features to numpy array
            features_df = np.asarray(features, dtype=np.float32)
            # print(np.shape(features))

            # create features dataset
            features_df = pd.DataFrame(features)
            features_df['category'] = dataset['category']
            features_df['time'] = dataset['time']
            features_df['invalid'] = dataset['censored'] == 1

            filename = f"{descriptor_name}_{'-'.join(map(lambda item : f'{item[0]}_{item[1]}', transform_combination.items()))}.csv"
            print(filename)
            print(features_df.shape)

            # save dataset to file as {transform_combination}_{descriptor_name}.csv
            features_df.to_csv(os.path.join(
                'features',  filename), index=False, header=False)
            features_df = None

            # concatenate features to full_features
            full_features = full_features + features
            # update max_full_feature_size
            max_full_feature_size = max(
                max_full_feature_size, max_feature_size)

        # pad each row in full_features to max_full_feature_size
        for (i, feature) in enumerate(full_features):
            full_features[i] = np.pad(feature, pad_width=(
                0, max_full_feature_size - len(feature)), mode='constant', constant_values=0)

        # convert full_features to numpy array
        full_features = np.asarray(full_features, dtype=np.float32)

        # create full_features dataset
        n_transform_combinations = len(transform_combinations)
        full_features = pd.DataFrame(full_features)
        full_features['category'] = list(
            dataset['category']) * n_transform_combinations
        full_features['time'] = list(
            dataset['time']) * n_transform_combinations
        full_features['invalid'] = list(
            dataset['censored'] == 1) * n_transform_combinations

        filename = f"{descriptor_name}_augmented.csv"
        print(filename)
        print(full_features.shape)
        # save dataset to file as {descriptor_name}_augmented.csv
        full_features.to_csv(os.path.join(
            'features', filename), index=False, header=False)

    except Exception as e:
        print(
            f"Error extracting features for {descriptor_name} {transform_combination}")
        print(e)
        error = True
