from data_cleaning import data_cleaning
from sklearn.model_selection import train_test_split
from ADAboost import adaboost_model
from knn import k_nearest_neighbors
from xgboost import xgboost

def main():
    data_cleaning()
    df= df
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=42)
    adaboost_model(x_train, x_test, y_train, y_test)

    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=4)
    k_nearest_neighbors(x_train, x_test, y_train, y_test)

    ##Change for bert
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=4)
    k_nearest_neighbors(x_train, x_test, y_train, y_test)

    ##Change for XGBoost
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=50)
    xg_boost(x_train, x_test, y_train, y_test)

    #Change for SGD
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=4)
    k_nearest_neighbors(x_train, x_test, y_train, y_test)
