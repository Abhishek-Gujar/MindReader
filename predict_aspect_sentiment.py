import math
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    review = []
    # Loading saved sentiment analysis model
    senti_clf = joblib.load(open('sentiment.pkl','rb'))

    # Loading saved sentiment analysis vectorizer model
    tf1 = joblib.load(open("vectorizer1.pkl", 'rb'))
    
    # Loding saved aspect analysis model
    aspect_clf = joblib.load(open('aspect.pkl','rb'))

    # Loading saved aspect analysis vectorizer model
    tf2 = joblib.load(open('vectorizer2.pkl', 'rb'))

    # Loading trained and saved SVM model
    svm = joblib.load(open('svm.pkl','rb'))

    def check_confidence(confidence):
        if confidence >=0 and confidence <=40:
            level = 'Low'
        elif confidence >40 and confidence <=70:
            level = 'Moderate'
        elif confidence >70:
            level = 'High'
        return level

    if request.method == 'POST':
        message = request.form['message']  
        review.append(message)     
        test = pd.DataFrame(columns=['string'])
        test = test.append({'string': ' '.join(review)}, ignore_index=True)

        # Transforming review using sentiment vectorizer model
        review1 = tf1.transform(test['string'])
        
        # Predicting sentiment for user input review
        sentiment_prediction = senti_clf.predict(review1)
        predicted_sentiment = sentiment_prediction[0]

        # Transforming review using aspect vectorizer model
        review2 = tf2.transform(test['string'])

        # Predicting aspect for user input review
        aspect_prediction = aspect_clf.predict(review2)
        predicted_aspect = aspect_prediction[0]
        
        # Computing confidence intervals
        confidence_int = svm.predict_proba(review1)

        # User review confidence measure
        if predicted_sentiment == 'Negative':
            confidence = math.ceil(confidence_int[0][0] * 100)
            level = check_confidence(confidence)
        elif predicted_sentiment == 'Neutral':
            confidence = math.ceil(confidence_int[0][1] * 100)
            level = check_confidence(confidence)
        elif predicted_sentiment == 'Positive':
            confidence = math.ceil(confidence_int[0][2] * 100)
            level = check_confidence(confidence)

    return render_template('result.html', predicted_sentiment=predicted_sentiment, predicted_aspect=predicted_aspect, confidence=confidence, message=message, level=level)

if __name__ == '__main__':
    app.run(debug=True)