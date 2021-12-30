import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K


class Predict_Bias:
    def __init__(self):
        return None

    @st.cache(allow_output_mutation=True)
    def get_model(self):
        saved_model = load_model("model.h5")    
        return saved_model
    
    def preprocess(self, text):
        new_text = text
        num_d_words = 50000
        maxlen = 300
        tokenizer = Tokenizer(num_words=num_d_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        new_text = tokenizer.texts_to_sequences(new_text)
        new_text = pad_sequences(new_text, maxlen=maxlen)
        #preprocessed = pad_sequences(new_text, maxlen=maxlen)
        return new_text

    def get_pred(self, text):
        model = self.get_model()
        pred = model.predict(self.preprocess(text))
        if pred >= 0.5:
            return str(f'This text is biased news with {pred[0][0]} certainty')
        else:
            return str(f'This text is balanced news with {pred[0][0]} certainty')




if __name__ == '__main__':
    st.title("Biased News Predictor")
    st.text("By Alan Reid | https://github.com/Alanapiereid")
    st.text("Trained on Keras LSTM")
    st.header("Is your news biased?")
    text = st.text_input('Enter a sentence to get a prediction')
    trigger = st.button('Get Prediction')

    if trigger:
        text_array = [text] 
        model = Predict_Bias()
        result = model.get_pred(text_array)
        st.text(result)

