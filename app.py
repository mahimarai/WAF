from flask import Flask, render_template, request
import pickle
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open('model_rfc.pkl', 'rb'))
vectorizer = pickle.load(open('Vect.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    a = request.form['Query']
    q= vectorizer.transform([a]) # a should be of ['abcd','bcdef']
    y = model.predict(q)
    if y==1:
    	d='Malicious'
    else:
    	d='Valid'
    return render_template('after.html', data=d, Query=a)


if __name__ == '__main__':
    app.run(debug=True)
    