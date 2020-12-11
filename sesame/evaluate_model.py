'''
This is a script to evaluate the accuracy of the model created earlier
'''
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from config import Config

# Loading in our test data for evaluation
x_test = pd.read_csv(str(Config.features_path / 'test_features.csv'))
y_test = pd.read_csv(str(Config.features_path / 'test_target.csv'))

# Loading in the trained model
model = pickle.load(open(str(Config.models_path / 'model.pickle'), 'rb'))

# Performing predictions on the trained model
y_pred = model.predict(x_test)

# Calculating metrics for the model (root Mean squared error and R²)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Saving our results in a JSON file
with open(str(Config.metrics_path), 'w') as outfile:
    json.dump(dict(r_squared=r2, rmse=rmse), outfile)
