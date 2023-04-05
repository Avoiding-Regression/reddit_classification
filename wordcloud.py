import matplotlib.pyplot as plt
from wordcloud import WordCloud

def wordcloud(data):
    all_words_lem = ' '.join([word for word in data['text_string_lem']])
    wcloud = WordCloud(width=600,
                       height=400,
                       max_font_size=100).generat(all_words_lem)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')