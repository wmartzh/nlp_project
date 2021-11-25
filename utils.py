from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import nltk
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from textblob import TextBlob
from wordcloud import WordCloud
plt.style.use('fivethirtyeight')
nltk.download('stopwords')


stop = set(stopwords.words('english'))


def remove_stop_words(text):
    filter_words = [word.lower()
                    for word in text.split()if word.lower() not in stop]
    return " ".join(filter_words)


def remove_reduntant_words(text):
    text = text.lower()
    text = re.sub('hotel', '', text)
    text = re.sub('room', '', text)
    text = re.sub('day', '', text)
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text


def get_subjetivity(text):

    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def polarity_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


def subjetivity_analysis(score):
    if score < 0:
        return 'Fact'
    else:
        return 'Opinion'


def plot_wordcloud(data):
    allWords = " ".join([word for word in data])
    wordCloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=119).generate(allWords)
    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def top_words(data):
    top_words = {}
    for c in data.columns:
        top = data[c].sort_values(ascending=False).head(30)
        top_words[c] = list(zip(top.index, top.values))
    return top_words


def most_common_words(data, top_words):
    words = []
    for item in data.columns:
        top = [word for (word, count) in top_words[item]]
        for t in top:
            words.append(t)
    return words



def plot_word_cloud(stop_words, data, title):
    allWords = " ".join([word for word in data])
    wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
                   max_font_size=150, random_state=42)
    wc.generate(allWords)
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title)
    plt.axis('off')

    plt.show()


def make_plot(data, string):
    plt.plot(data)

    plt.show()


if __name__ == '__main__':
    pass
