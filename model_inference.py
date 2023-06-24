from data_cleaning import data_cleaning
from sklearn.model_selection import train_test_split
from ADAboost import adaboost_model
from knn import k_nearest_neighbors
from xgboost import xgboost
from data_cleaning import data_cleaning
from sklearn.model_selection import train_test_split
from ADAboost import adaboost_model
from knn import k_nearest_neighbors
from sgd import sgd
import data_generation 

def model_inf(final_df):
    df = final_df
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=42)
    adaboost_model(x_test, y_test)

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=4)
    k_nearest_neighbors(x_test, y_test)


    ##Change for XGBoost
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=50)
    xgboost(x_test, y_test)


    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=4)
    sgd(x_test, y_test)

    return adaboost_model, k_nearest_neighbors,xgboost, sgd
