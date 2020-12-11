'''
This is a script to extracting and preprocessing features
from the data provided
'''
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import Config

# Creating path to store features extracted
Config.features_path.mkdir(parents=True, exist_ok=True)

# Reading in our training and testing data into pandas DataFrame
train_df = pd.read_csv(str(Config.dataset_path / 'train.csv'))
test_df = pd.read_csv(str(Config.dataset_path / 'test.csv'))

# Creating a function to extract features and preprocess them


def feature_extraction(dframe):
    '''
    Function for extracting and preprocessing features
    '''
    features = dframe[['Area harvested', 'Yield']]
    scale = StandardScaler()
    preprocessed = scale.fit_transform(features)
    return preprocessed


train_features = feature_extraction(train_df)
test_features = feature_extraction(test_df)

# Saving the preprocessed features to our path
pd.DataFrame(train_features, columns=[ 'Area harvested', 'Yield']).to_csv(
    str(Config.features_path / 'train_features.csv'), index=None
)
pd.DataFrame(test_features, columns=[ 'Area harvested', 'Yield']).to_csv(
    str(Config.features_path / 'test_features.csv'), index=None
)

# Creating a function to preprocess our target variable


def preprocess_target(dframe):
    '''
    Function for extracting and preprocessing the target
    '''
    features = dframe['Production'].values.reshape(-1, 1)
    scale = StandardScaler()
    return scale.fit_transform(features)


train_target = preprocess_target(train_df)
test_target = preprocess_target(test_df)

# Saving the preprocessed target to our path
pd.DataFrame(train_target, columns=[['Production']]).to_csv(
    str(Config.features_path / 'train_target.csv'), index=None
)
pd.DataFrame(test_target, columns=['Production']).to_csv(
    str(Config.features_path / 'test_target.csv'), index=None
)
