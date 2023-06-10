from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def sgd(x_train, y_train):
    sgd_clf = SGDClassifier()
    sgd_clf.fit(x_train, y_train)
    return sgd_clf


