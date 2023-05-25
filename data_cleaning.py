nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

def data_cleaning():
    gs = pd.read_csv('gamestop.csv')
    tesla = pd.read_csv('tesla.csv')
    gs['text'] = gs['title'].asstype(str).str.lower()
    tesla['text'] = tesla['title'].asstype(str).str.lower()
    regexp = RegexpTokenizer('\w+')

    gs['text_token'] = gs['text'].apply(regexp.tokenize)
    tesla['text_toke'] = tesla['text'].apply(regexp.tokenize)
    stopwords = nltk.corpus.stopwords.words('english')
    newStopWords = ['megathread', 'Megathread', 'daily', '2022', 'thread', 'Thread', 'Daily']
    stopwords.extend(newStopWords)

    gs['text_token'] = gs['text_token'].apply(lambda x:[item for item in x if item not in stopwords])
    tesla['text_token'] = tesla['text_token'].apply(lambda x:[item for item in x if item not in stopwords])
    
    gs['text_string'] = gs['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))
    tesla['text_string'] = tesla['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))

    all_words_gs = ' '.join([word for word in gs['text_string']])
    all_words_tesla = ' '.join([word for word in tesla['text_string']])

    tokenized_words_gs = nltk.tokenize.word_tokenize(all_words_gs)
    tokenized_words_tesla = nltk.tokenize.word_tokenize(all_words_tesla) 

    fdist_gs = FreqDist(tokenized_words_gs)
    fdist_tesla = FreqDist(tokenized_words_tesla)

    gs['text_string_fdist'] = gs['text_token'].apply(lambda x: ' '.join([item for item in x if fdist_gs[item] >= 1]))
    tesla['text_string_fdist'] = tesla['text_token'].apply(lambda x: ' '.join([item for item in x if fdist_tesla[item] >= 1]))

    wordnet_lem_gs= WordNetLemmatizer()
    wordnet_lem_tesla= WordNetLemmatizer()

    gs['text_string_lem'] = gs['text_string_fdist'].apply(wordnet_lem_gs.lemmatize)
    tesla['text_string_lem'] = tesla['text_string_fdist'].apply(wordnet_lem_tesla.lemmatize)

    frames = [tesla, gs]
    df = pd.concat(frames)


    df['labels'] = np.where(df['subreddit'] == 'teslainvestorsclub', 1, 0)
    df['text'] = df['text_string_lem']

    return df
