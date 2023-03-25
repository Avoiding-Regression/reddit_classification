import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

regexp = RegexpTokenizer('\w+')
wordnet_lem= WordNetLemmatizer()

gamestop = pd.read_csv('gamestop.csv', index_col=0)
tesla = pd.read_csv('tesla.csv', index_col=0)

print(gamestop.isnull().sum())
print(tesla.isnull().sum())

tesla['text_length'] = tesla['title'].apply(len)
sns.histplot(data=tesla, x = 'text_length')
plt.title('Tesla Text Length')
plt.xlabel('Text Length')
plt.ylabel('Text Count')
plt.show()

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['megathread', 'Megathread', 'daily', '2022', 'thread', 'Thread', 'Daily']
stopwords.extend(newStopWords)


tesla['text'] = tesla['title'].astype(str).str.lower()
tesla['text_token'] = tesla['text'].apply(regexp.tokenize)
tesla['text_token'] = tesla['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
tesla['text_string'] = tesla['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>1]))

all_tesla_words = ' '.join([word for word in tesla['text_string']])
tokenized_tesla_words = nltk.tokenize.word_tokenize(all_tesla_words)
tesla_fdist = FreqDist(tokenized_tesla_words)


tesla['text_string_fdist'] = tesla['text_token'].apply(lambda x: ' '.join([item for item in x if tesla_fdist[item] >= 1]))
tesla['text_string_lem'] = tesla['text_string_fdist'].apply(wordnet_lem.lemmatize)
tesla['is_equal'] = (tesla['text_string_fdist']== tesla['text_string_lem'])

print(tesla.is_equal.value_counts())

all_tesla_words_lem = ' '.join([word for word in tesla['text_string_lem']])

tesla_wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_tesla_words_lem)

plt.figure(figsize=(10, 7))
plt.imshow(tesla_wordcloud, interpolation='bilinear')
plt.axis('off');
plt.title('Tesla Word Cloud')
plt.show()

tesla_words = nltk.word_tokenize(all_tesla_words_lem)
fd = FreqDist(tesla_words)
tesla_top_10 = fd.most_common(10)
tesla_fdist = pd.Series(dict(tesla_top_10))
sns.set_theme(style = "ticks")
sns.barplot(y=tesla_fdist.index, x=tesla_fdist.values, color = 'green')
plt.show()