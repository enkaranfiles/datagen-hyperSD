import pandas as pd
import pyarrow.parquet as pq
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data):
        pass

class ParquetReader(DataProcessor):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def process(self, data=None):
        dfs = []
        for file_path in self.file_paths:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            dfs.append(df)
        return pd.concat(dfs)

class DataFilter(DataProcessor):
    def __init__(self, width_threshold, height_threshold):
        self.width_threshold = width_threshold
        self.height_threshold = height_threshold

    def process(self, data):
        return data[(data['WIDTH'] > self.width_threshold) & (data['HEIGHT'] > self.height_threshold)]

class DataFrameParquetWriter(DataProcessor):
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, data):
        data.to_parquet(self.output_path)
        return data

class DataPipeline:
    def __init__(self, processors):
        self.processors = processors

    def execute(self):
        data = None
        for processor in self.processors:
            data = processor.process(data)
        return data

file_paths = ['parquet-files/train-00000-of-00007-29aec9150af50f9f.parquet',
            'parquet-files/train-00001-of-00007-060633a36bcf0956.parquet', 
            'parquet-files/train-00002-of-00007-709151a2715d894d.parquet',
            'parquet-files/train-00003-of-00007-2dc95366d4278bb8.parquet', 
            'parquet-files/train-00004-of-00007-f06fcc8b41bf5fdf.parquet', 
            'parquet-files/train-00005-of-00007-f1ec12a5b5b3e6c0.parquet',
            'parquet-files/train-00006-of-00007-57e09e020b9c2df4.parquet']

processors = [
    ParquetReader(file_paths),
    DataFilter(512, 512),
    DataFrameParquetWriter('filtered_data.parquet')
]

pipeline = DataPipeline(processors)
filtered_data = pipeline.execute()
