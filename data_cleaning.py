import pandas as pd
import numpy as np 
import data_generation as dg
import data_helper as dh
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# need to bring in the data here dg.main()
def data_cleaning(datasource):
    datasource['text'] = datasource['title'].astype(str).str.lower()
    regexp = RegexpTokenizer('\w+')
    datasource['text_token'] = datasource['text'].apply(regexp.tokenize)
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['megathread', 'Megathread', 'daily', '2022', 'thread', 'Thread', 'Daily']
    stopwords.extend(newStopWords)
    datasource['text_token'] = datasource['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
    datasource['text_string'] = datasource['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))
    all_words = ' '.join([word for word in datasource['text_string']])
    tokenized_words = nltk.tokenize.word_tokenize(all_words)

    return tokenized_words, datasource



