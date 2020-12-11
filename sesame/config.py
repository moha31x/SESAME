'''
This is a script to provide configuration dependencies for the entire project
'''
from pathlib import Path


class Config:
    '''
    This is a class object to store in configuration details
    '''
    random_seed = 42
    assets_path = Path('./assets')
    original_data_path = assets_path / 'original_data' / 'dataset.csv'
    dataset_path = assets_path / 'data'
    features_path = assets_path / 'features'
    models_path = assets_path / 'models'
    metrics_path = assets_path / 'metrics.json'
