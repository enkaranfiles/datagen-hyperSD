import matplotlib.pyplot as plt

from datagen import Context, CannyStrategy, DepthStrategy
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import urllib.request

import re


def read_image_from_url(image_url):
    with urllib.request.urlopen(image_url) as url_response:
        image_array = np.asarray(bytearray(url_response.read()), dtype=np.uint8)
    return cv2.imdecode(image_array, -1)

def sanitize_filename(name):
    '''
    added prompt sentence the image name head, so imwrite gives error! 
    '''
    pattern = '[^a-zA-Z0-9_-]'
    return re.sub(pattern, '', name)

# Modify the save_image function accordingly
def save_image(image, image_name, folder):
    sanitized_name = sanitize_filename(image_name)
    if 'depth' in folder:
        # Normalize the depth image to 0-255 range for visualization
        normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(f'{folder}/{sanitized_name}.jpg', normalized_image)
    else:
        cv2.imwrite(f'{folder}/{sanitized_name}.jpg', image)

def process_and_save_images(df):
    context = Context(CannyStrategy())
    for index, data in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            image = read_image_from_url(data['URL'])
            save_image(image, f'image_{data["TEXT"]}', 'input_image')

            canny_image = context.process_image(image)
            save_image(canny_image, f'canny_image_{data["TEXT"]}', 'condition/canny')

            context.strategy = DepthStrategy()
            depth_image = context.process_image(image)
            save_image(depth_image, f'depth_image_{data["TEXT"]}', 'condition/depth')
        except Exception as e:
            print(f'Error for index {index}, error message: {e}')

if __name__ == '__main__':
    df = pd.read_parquet('filtered_data.parquet', engine='pyarrow')
    df.reset_index(drop=True, inplace=True)
    process_and_save_images(df)
