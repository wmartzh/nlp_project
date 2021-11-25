import pandas as pd
import numpy as np
import utils as utl
import pickle
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

np.set_printoptions(precision=3, suppress=True)


# Load Data
data = pd.read_csv('./data/tripadvisor_hotel_reviews.csv')


data['Review'] = data.Review.apply(utl.clean_text)

print(data.head(20))
clean_data = data.head(100)
data = data.head(100)


clean_data.to_pickle("./data/corpus.pkl")

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(clean_data.Review)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = clean_data.index

# Clean data & generate new files
data_dtm.to_pickle("./data/dtm.pkl")
clean_data.to_pickle('./data/data_clean.pkl')
pickle.dump(cv, open("./data/cv.pkl", "wb"))

dtm = pd.read_pickle('./data/dtm.pkl')
dtm = dtm.transpose()


top_words = utl.top_words(dtm)
words = utl.most_common_words(dtm, top_words)


# print(words)
add_stop_words = [word for word, count in Counter(
    words).most_common() if count > 6]
# 
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# print(data)
# utl.plot_word_cloud(stop_words, clean_data.Review.values, 'Most common word')


data['Polarity'] = data.Review.apply(utl.get_polarity)
data['Subjetivity'] = data.Review.apply(utl.get_subjetivity)
data['Opinion'] = data.Polarity.apply(utl.polarity_analysis)
data['Truth'] = data.Subjetivity.apply(utl.subjetivity_analysis)


positive_filter = data["Opinion"] == 'Positive'
negative_filter = data["Opinion"] == 'Negative'


positive_opinions = data.where(positive_filter, inplace=False)
negative_opinions = data.where(negative_filter, inplace=False)
positive_opinions = positive_opinions.dropna()
negative_opinions = negative_opinions.dropna()
# print("Positive\n")
# print(positive_opinions)
# print("Negative \n")
# print(negative_opinions)

# # Get data
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(negative_opinions.Review)
dtm_negative = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
dtm_negative.index = negative_opinions.index
# print(dtm_negative)
cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(positive_opinions.Review)
dtm_postivie = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
dtm_postivie.index = positive_opinions.index


# top_positive = utl.top_words(dtm_postivie)
# top_negative = utl.top_words(dtm_negative)

# print(top_positive)
# print(top_negative)
utl.plot_word_cloud(stop_words, negative_opinions.Review.values, 'Negative')
utl.plot_word_cloud(stop_words, positive_opinions.Review.values, 'Positive')


