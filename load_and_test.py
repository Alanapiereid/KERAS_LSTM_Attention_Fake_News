import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
import pickle

num_d_words = 50000

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

maxlen = 300



saved_model = load_model('fake_n_model.h5')
new_text = np.array(['Wow Trump has really gone off the deepend this time and very few people will be surprised by that'])
new_text2 = np.array(['Speculation grows that Maxwell may try to cut a deal for reduced sentence'])
new_text3 = np.array(['According to Lichtman, there are defendants who, in the eyes of the government, are so bad that it does not want to strike a deal in exchange for testimony.'])
newer = np.array(['WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a â€œfiscal conservativeâ€ on Sunday and urged budget restraint in 2018. In keeping with a sharp pivot under way among Republicans, U.S. Representative Mark Meadows, speaking on CBSâ€™ â€œFace the Nation,â€ drew a hard line on federal spending, which lawmakers are bracing to do battle over in January.'])
# apply tokenization + padding, otherwise this won't work

def preprocess_and_pred(text):
    new_text = tokenizer.texts_to_sequences(text)
    new_text = pad_sequences(new_text, maxlen=maxlen)
    pred = saved_model.predict(new_text)
    if pred >= 0.5:
        return f'This text is biased news with {pred[0]} certainty'
    else:
        return f'This text is balanced news with {pred[0]} certainty'


print(preprocess_and_pred(new_text))
print(preprocess_and_pred(new_text2))
print(preprocess_and_pred(new_text3))
print(preprocess_and_pred(newer))