import xgboost as xgb
import numpy as np

def xgboost():

# Setting the Parameters of the Model
    param = {'eta': 0.75,
            'max_depth': 50,
            'objective': 'binary:logitraw'}
# Training the Model
    xgb_model = xgb.train(param, xgb_train, num_boost_round = 30)
    return xgb_model