# SESAME
Sesame-Production
Codacy Badge

Introduction
The data used in this workflow has been obtained from the United Nations Food and Agriculture Organisation public data repository.

The objective of this ML Workflow is to accurately predict the Production quantity of sesame seeds provided the Area harvested and the Yield Expected.

This steps of this workflow consists of
Extracting the data from a remote storage.

Feature extraction, preprocessing and transformation.

A machine learning algorithm is applied to the transformed data and saved in a serialized format.

Finally the last step of our pipeline is model evaluation.

The metrics used for this project are
Coefficent of Determination (r²)
Root Mean Squared Error (rmse)
