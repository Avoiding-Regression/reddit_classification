import xgboost as xgb
import numpy as np

def xgboost():

# Setting the Parameters of the Model
    param = {'eta': 0.75,
            'max_depth': 50,
            'objective': 'binary:logitraw'}
# Training the Model
    xgb_model = xgb.train(param, xgb_train, num_boost_round = 30)
# Predicting using the Model
    y_pred = xgb_model.predict(xgb_test)
    y_pred = np.where(np.array(y_pred) > 0.5, 1, 0) # converting them to 1/0’s