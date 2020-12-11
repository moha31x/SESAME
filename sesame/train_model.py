'''
This is a script to train a model with a variety of estimators
'''
import pickle
import pandas as pd
from sklearn.neural_network import MLPRegressor
from config import Config

# Creating a path to save our model
Config.models_path.mkdir(parents=True, exist_ok=True)

# Loading the training and testing features into a pandas DataFrame
x_train = pd.read_csv(str(Config.features_path / 'train_features.csv'))
y_train = pd.read_csv(str(Config.features_path / 'train_target.csv'))

# Instantiating and fitting the algorithm
model = MLPRegressor(max_iter=800, alpha=0.4371)
model = model.fit(x_train, y_train.to_numpy().ravel())

# Saving the model into a pickle file
pickle.dump(model, open('model.pickle', 'wb'))
