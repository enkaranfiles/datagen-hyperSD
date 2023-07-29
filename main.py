
from datagen import Context, CannyStrategy, HEDStrategy, DepthStrategy
from create_data import ParquetReader
import pandas as pd
if __name__ == '__main__':
    file_path = 'filtered_data.parquet'
    print('File Reader...')
    dataframe = ParquetReader('filtered_data.parquet').process()
    print(dataframe.columns)
    #context = Context(CannyStrategy())


    #context.process_image('image_path')

    #context.strategy = HEDStrategy()
    #context.process_image('image_path')

    #context.strategy = DepthStrategy()
    #context.process_image('image_path')