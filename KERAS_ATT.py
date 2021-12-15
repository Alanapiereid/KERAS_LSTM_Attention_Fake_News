import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

# df_1 = pd.read_csv('DEFAM_SET.csv', encoding='utf-8')
# print(len(df_1))
# df_2 = pd.read_csv('NEUT_SET.csv', encoding='utf-8')
# def_data = df_1[:25000]
# neut_data = df_2[:25000]
# comb_frames = [def_data, neut_data]
# df = pd.concat(comb_frames)
# df = df.sample(frac=1).reset_index(drop=True)

df_1 = pd.read_csv('fn_train.csv')
df_2 = pd.read_csv('fn_test.csv')


def data_clean(df):
    df = df.drop(columns=['id', 'title', 'author'])
    df = df.loc[(df["text"].notnull()) & df["label"].notnull()]

    return df
#print(data_clean(df_1).head)

df_1 = data_clean(df_1)

tokenizer = Tokenizer(num_words=50000, oov_token='<UNK>')
# Model configuration
additional_metrics = ['accuracy']
batch_size = 128
embedding_out_dim = 100
loss_function = BinaryCrossentropy()
maxlen = 300
num_d_words = 50000
number_of_epochs = 2
optimizer = Adam()
validation_split = 0.20
verbosity_mode = 1

texts = df_1['text']
labels = df_1['label']

# Load dataset
# x_train, x_test, y_train, y_test = train_test_split(
#     texts, labels, stratify=labels,
#     random_state=0)

tokenizer = Tokenizer(num_words=num_d_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# crucial
tokenizer.fit_on_texts(texts.values)
word_index = tokenizer.word_index

#crucial
X = tokenizer.texts_to_sequences(texts.values)
X = pad_sequences(X, maxlen=maxlen)
y = labels.values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, stratify=labels,
    random_state=0)

print(x_train.shape)
print(x_test.shape)


# Define the Keras model
model = Sequential()
model.add(Embedding(num_d_words, embedding_out_dim, input_length=maxlen))
model.add(LSTM(10))
model.add(Dense(1, activation='sigmoid'))

# # Compile the model
model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)

# # Give a summary
model.summary()

# # Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# # Test the model after training
results = model.evaluate(x_test, y_test, verbose=False)
print(f'Results - Loss: {results[0]} - Accuracy: {100*results[1]}%')