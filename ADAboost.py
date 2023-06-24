from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def adaboost_model(train_x, test_x, y_train, y_test):
    cv = CountVectorizer(binary = True)
    #cv.fit(final_df['text'])#don't think we need this, but we need have all the inputs be uniform 
    train_x = cv.transform(train_x)
    test_x = cv.transform(test_x)

    abc=AdaBoostClassifier(n_estimators = 50
                           learning_rate = 1)
    abc.fit(train_x, test_x)
    
    return abc
    

    
