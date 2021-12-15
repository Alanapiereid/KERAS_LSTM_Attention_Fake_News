import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import model_from_json


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
h5_file_name = 'model.h5'

st.title("Keras Prediction Basic UI")
st.header("A Streamlit based Web UI To Get Predictions From Trained Models")

# load model from json
json_file = open('model.json', 'r')
saved_model_json = json_file.read()
json_file.close()
saved_model = model_from_json(saved_model_json)
# load weights from h5 file
saved_model.load_weights("model.h5")
print("Loaded")

# compile and test saved model
saved_model.compile(loss=loss_function, optimizer=optimizer, metrics=additional_metrics)
score = saved_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (saved_model.metrics_names[1], score[1]*100))

# get prediction for new text
new_text = np.array(['Wow Trump has really gone off the deepend this time and very few people will be surprised by that'])
# apply tokenization + padding, otherwise this won't work
new_text = tokenizer.texts_to_sequences(new_text)
new_text = pad_sequences(new_text, maxlen=maxlen)
# make a prediction
pred = saved_model.predict(new_text)
# print result
if pred >= 0.5:
    print(f'This text is biased news with {pred[0]} certainty')
else:
    print(f'This text is balanced news with {pred[0]} certainty')

