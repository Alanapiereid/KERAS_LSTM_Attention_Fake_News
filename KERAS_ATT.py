import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout1D
import pickle


# obtain the original file (login/api credentials needed) from https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
# there is some data wrangling to be done
df1 = pd.read_csv('Fake.csv', dtype={'title': 'str', 'text': 'str'})
df2 = pd.read_csv('True.csv', dtype={'title': 'str', 'text': 'str'})
df1 = df1[['title','text']]
df2 = df2[['title','text']]
# I add binary labels here
df1['label'] = 1
df2['label'] = 0
# this makes sure the positive/negative datasets are of equal size before merging
frame1 =df1[:len(df2)]
use_frame = frame1.append(df2, ignore_index=True)
# merge title and text columns into one feature column
use_frame["sentence"] = use_frame["title"] + use_frame["text"]
use_frame = use_frame.drop(['title', 'text'], axis=1)
cols = use_frame.columns.tolist()
cols = [cols[1] , cols[0]]
use_frame = use_frame[cols]
#print(use_frame.head)
##################################################
# # Model params
additional_metrics = ['accuracy']
batch_size = 64
embedding_out_dim = 100
loss_function = BinaryCrossentropy()
maxlen = 250
num_d_words = 50000
number_of_epochs = 5
optimizer = Adam(learning_rate=0.001)
validation_split = 0.20
verbosity_mode = 1
#################################################
# ###############################################
texts = use_frame['sentence']
labels = use_frame['label']
# ##############################################
# # define tokenizer
tokenizer = Tokenizer(num_words=num_d_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# # fit tokenizer to texts
tokenizer.fit_on_texts(texts.values)
word_index = tokenizer.word_index
# # turn tokenized texts into sequences to preserve word order
X = tokenizer.texts_to_sequences(texts.values)
# # pad for lengths
X = pad_sequences(X, padding='post', maxlen=maxlen, truncating='post')
# # set y values
y = labels.values

# # train/test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=0)

# print(x_train.shape)
# print(x_test.shape)

accuracy_callback = 0.90

class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get('accuracy') > accuracy_callback):
			print("\nReached %2.2f%% accuracy - training over" %(accuracy_callback*100))
			self.model.stop_training = True

# Instantiate callback
callbacks = myCallback()

# # Define Keras model
model = Sequential()
model.add(Embedding(num_d_words, embedding_out_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))

# Compile
model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

# Summary
model.summary()

# Train
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split, callbacks=[callbacks])

# Test
results = model.evaluate(x_test, y_test, verbose=False)
print(f'Results - Loss: {results[0]} - Accuracy: {100*results[1]}%')
# save model
model.save('fake_n_model.h5')
# load model
#saved_model = load_model('fake_n_model.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

##########################################################################################

# compile and test saved model
# saved_model.compile(loss=loss_function, optimizer=optimizer, metrics=additional_metrics)
# score = saved_model.evaluate(X, y, verbose=0)
# print("%s: %.2f%%" % (saved_model.metrics_names[1], score[1]*100))

# get prediction for new text
new_text = np.array(['Wow Trump has really gone off the deepend this time and very few people will be surprised by that'])
new_text2 = np.array(['Speculation grows that Maxwell may try to cut a deal for reduced sentence'])
new_text3 = np.array(['According to Lichtman, there are defendants who, in the eyes of the government, are so bad that it does not want to strike a deal in exchange for testimony.'])
# apply tokenization + padding, otherwise this won't work

def preprocess_and_pred(text):
    new_text = tokenizer.texts_to_sequences(text)
    new_text = pad_sequences(new_text, maxlen=maxlen)
    pred = model.predict(new_text)
    if pred >= 0.5:
        return f'This text is biased news with {pred[0]} certainty'
    else:
        return f'This text is balanced news with {pred[0]} certainty'


print(preprocess_and_pred(new_text))
print(preprocess_and_pred(new_text2))
print(preprocess_and_pred(new_text3))

