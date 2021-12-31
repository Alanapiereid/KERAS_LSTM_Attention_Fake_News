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
import pickle


class Predict_Bias:
    def __init__(self):
        self.new_text = None

    @st.cache(allow_output_mutation=True)
    def get_model(self):
        saved_model = load_model("fake_n_model.h5")    
        return saved_model
    
    def preprocess(self, text):
        new_text = text
        num_d_words = 50000
        maxlen = 300
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        new_text = tokenizer.texts_to_sequences(new_text)
        self.new_text = pad_sequences(new_text, maxlen=maxlen)
        #preprocessed = pad_sequences(new_text, maxlen=maxlen)
        return self.new_text

    def get_pred(self, text):
        model = self.get_model()
        pred = model.predict(self.preprocess(text))
        if pred >= 0.5:
            return str(f'This text is biased news with {pred[0][0]} certainty')
        else:
            return str(f'This text is balanced news with {pred[0][0]} certainty')




if __name__ == '__main__':
    st.title("Biased News Article Predictor")
    st.text("By Alan Reid | https://github.com/Alanapiereid")
    st.text("Trained on Keras LSTM")
    st.header("Is your news biased?")
    text = st.text_input('Paste a news article into the field below to get a prediction')
    text_array = [text] 
    trigger = st.button('Get Prediction')
    model = Predict_Bias()
    if trigger:
        result = model.get_pred(text_array)
        st.text(result)

