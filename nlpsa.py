import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import utils as utl
np.set_printoptions(precision=3, suppress=True)

# LOAD DATA
data = pd.read_csv('./data/tripadvisor_hotel_reviews.csv')
data.colums = ['Review', 'Rating']
# print(data.Review.head())

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"


# CLEAN DATA
data['Review'] = data.Review.apply(utl.clean_text)
data['Review'] = data.Review.apply(utl.remove_reduntant_words)
data["Review"] = data.Review.apply(utl.remove_stop_words)

data["Polarity"] = data.Review.apply(utl.get_polarity)
data["Subjectivity"] = data.Review.apply(utl.get_subjetivity)
data["PolarityAnalysis"] = data.Polarity.apply(utl.polarity_analysis)
print(data)



# SET TRAINING VALUE
training_size = int(data.shape[0] * 0.8)
# SPLIT LABELS AND SENTENCES
sentences = []
labels = []

sentences = []
labels = []

for item in data.Review.values:

    sentences.append(item)

for item in data.Rating.values:
    labels.append(item)


train_sentences = sentences[0:training_size]
test_sentences = sentences[training_size:]
train_labels = labels[0:training_size]
test_labels = labels[training_size:]

# GET TOKENS



tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
words_index = tokenizer.word_index


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(
    train_sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

utl.plot_wordcloud(data.Review)

print(train_padded)


test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, padding=padding_type,
                            maxlen=max_length, truncating=trunc_type)

training_padded = np.array(train_padded)
training_labels = np.array(train_labels)
testing_padded = np.array(test_padded)
testing_labels = np.array(test_labels)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
print(model.summary())
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)


# print(words_index)
# print(padded[0])
