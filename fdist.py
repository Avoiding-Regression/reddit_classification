from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import pandas as pd
import seaborn as sns

def fdist(data):
    all_words_lem = ' '.join([word for word in data['text_string_lem']])
    words = nltk.word_tokenize(all_words_lem)
    fd = FreqDist(words)
    top_10 = fd.most_common(10)
    fdist = pd.Series(dict(top_10))
    sns.set_theme(style = "ticks")
    sns.barplot(y=fdist.index, x=fdist.balues, color = 'blue')
