from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

def k_nearest_neighbors(x_train, x_test, y_train, y_test): 
    vectorizer = CountVectorizer(ngram_range=(1,2))
    train_bow = vectorizer.fit_transform(x_train.values)
    test_bow = vectorizer.transform(x_test.values)
    # feature_names_bow = vectorizer.get_feature_names() I don't think we need this?
    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(train_bow, test_bow)
    return model

    # need additional steps for inference
    