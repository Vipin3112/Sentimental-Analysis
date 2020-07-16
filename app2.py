
import pandas as pd
import numpy as np
from flask import Flask,render_template,url_for,request
import pickle
from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import bz2
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    model = pickle.load(open('NB_model.pkl', 'rb'))
    cv=pickle.load(open('cv', 'rb'))
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer

    if request.method == 'POST':
        message = request.form['message']

        data=[]
        review = re.sub("<.*?>", "", message)
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = re.sub(r'[^\w\s2]',' ',review)
        review = re.sub('<br /><br />'," ",review)
        review = review.lower()
        review = review.split()
        
        lemmatizer = WordNetLemmatizer() 
        review = [lemmatizer.lemmatize(words) for words in review if not words in set(stopwords.words('english'))]
        review = ' '.join(review)
        data.append(review)

        # vect = cv.transform(data).toarray()
        my_prediction = model.predict(cv.transform(data).toarray())
        if my_prediction==0:
            return render_template('neg.html')
        else:
            return render_template('pos.html')
        
    


if __name__ == '__main__':
    app.run(debug=True)