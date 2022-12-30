import os.path
from pprint import pprint
import time
from io import BytesIO
from random import random
import uuid
from PIL import Image
from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
import azure.cognitiveservices.vision.contentmoderator.models
from msrest.authentication import CognitiveServicesCredentials
import pandas as pd
import sys

CONTENT_MODERATOR_ENDPOINT = "https://researchmoderator.cognitiveservices.azure.com/"
subscription_key = "40d6e22d021441fba70bbc6de364b000"

client = ContentModeratorClient(
    endpoint=CONTENT_MODERATOR_ENDPOINT,
    credentials=CognitiveServicesCredentials(subscription_key)
)

# print("\nEvaluate for adult and racy content.")
# evaluation = client.image_moderation.evaluate_file_input(open(
#     './underwear/4skins   Men and underwear.jpg', 'rb'))
# # assert isinstance(evaluation, Evaluate)
# pprint(evaluation.as_dict())

categories = {
    'invalid': {
        'folder': 'invalid',
        'class': 0,
        'evaluate': True
    },
    'underwear': {
        'folder': 'underwear',
        'class': 1,
        'evaluate': True
    },
    'valid': {
        'folder': 'valid',
        'class': 1,
        'evaluate': False
    }
}

# initialize pandas data frame with columns: file_path, adult_score, racy_score, mean_score, category, class
dataset = pd.read_csv('image_scores.csv')

for (category, props) in categories.items():
    # iterate over corresponding folder for each image category and evaluate image
    cwd = os.path.dirname(os.path.realpath(__file__))
    folder_path = os.path.join(cwd, props['folder'])
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if props['evaluate']:
            # retry evaluation up to 3 times if error occurs
            attempts = 0
            while attempts < 3:
                try:
                    evaluation = client.image_moderation.evaluate_file_input(
                        open(file_path, 'rb'))
                    break
                except Exception as e:
                    attempts += 1
                    time.sleep(1)
                    pprint(e)

            # append row to dataset
            dataset = dataset.append({
                'file_path': file_path,
                'adult_score': evaluation.adult_classification_score,
                'racy_score': evaluation.racy_classification_score,
                'mean_score': (evaluation.adult_classification_score + evaluation.racy_classification_score) / 2,
                'category': category,
                'class': props['class']
            }, ignore_index=True)
        else:
            dataset = dataset.append({
                'file_path': file_path,
                'adult_score': 0,
                'racy_score': 0,
                'mean_score': 0,
                'category': category,
                'class': props['class']
            }, ignore_index=True)

        # break
        dataset.to_csv(os.path.join(cwd, 'image_scores.csv'), index=False)

# sort dataset by mean score
# dataset = dataset.sort_values(by=['mean_score'], ascending=False)
