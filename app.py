from flask import Flask, request, render_template
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
from flask import jsonify

model = 'LSTM'

# Declare a Flask app
app = Flask(__name__)

saved_model = load_model('fake_n_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess_and_pred(text):
    text = [text]
    new_text = tokenizer.texts_to_sequences(text)
    new_text = pad_sequences(new_text, maxlen=300)
    pred = saved_model.predict(new_text)
    if pred >= 0.5:

        return 'biased', str(pred[0][0])
    else:
        return 'balanced', str(pred[0][0])

# Main function here
@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
               
        # Get values through input bars
        article = request.form.get("article")
        print(type(article))
        # Get prediction
        prediction, prob_ = preprocess_and_pred(article)

        result = {
        "text": article,
        "output": {
            'prediction': prediction,
            'probability': prob_
        },
        "meta": {
            "model": model
        }}
        return result   
        
    elif request.method == "GET":
               return render_template("front_end.html")
    

# Running the app
if __name__ == '__main__':
    app.run(debug = True)