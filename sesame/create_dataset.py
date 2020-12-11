'''
This is a script to create and obtain the data from remote storage
'''
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gdown
from config import Config

# Setting the random seed generator
np.random.seed(Config.random_seed)
random = Config.random_seed

# Creating file path for saving the dataset
Config.original_data_path.parent.mkdir(parents=True, exist_ok=True)
Config.dataset_path.mkdir(parents=True, exist_ok=True)

# Downloading the data from Google Drive
gdown.download(
    os.environ.get('DATA'),
    str(Config.original_data_path)
)

# Reading in our dataset into pandas DataFrame
df = pd.read_csv(str(Config.original_data_path), encoding='latin1')

# Splitting our data into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2,
                                     random_state=random)

# Saving our splitted data to the path we created earlier
df_train.to_csv(str(Config.dataset_path / 'train.csv'), index=None)
df_test.to_csv(str(Config.dataset_path / 'test.csv'), index=None)
