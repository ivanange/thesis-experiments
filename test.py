from PIL import Image
import numpy as np
from kernel_descriptors_extractor import KernelDescriptorsExtractor
from sift_feature_extractor import SIFTFeatureExtractor
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage.color import rgb2gray
from skimage import io

# load the image and convert into
# numpy array
image = Image.open(
    '../images/dataset/invalid\I felt really proud of a full frontal nude for the first time in a while  i  hope you like it🥺   r RealGirls.jpg')

print(image.format)
print(image.size)
print(image.mode)

# asarray() class is used to convert
# PIL images into NumPy arrays
# numpydata = np.array(image)
numpydata = io.imread(
    '../images/dataset/invalid\I felt really proud of a full frontal nude for the first time in a while  i  hope you like it🥺   r RealGirls.jpg')
# <class 'numpy.ndarray'>
print(type(numpydata))

#  shape
print(numpydata.shape)

# sift = SIFT()
# gray = rgb2gray(image)
# print(gray.shape)
# sift.detect_and_extract(gray)
# sift_features = sift.descriptors
# print(sift_features)
# print(sift.keypoints)


# kdes = KernelDescriptorsExtractor()
# print(numpydata)
# kdes_features = kdes.predict(np.array([numpydata]))
# print(kdes_features)
