import math
import joblib
import warnings
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

def create_sentiment_dataframe():
    data = pd.read_csv('dataset.csv', low_memory=False, encoding='ISO-8859-1')
    data = data[['primaryCategories','reviews.rating','reviews.text']]
    # print(data.isnull().sum())
    # print(data.shape)
    return data

def stopwords_filter(df):
    words = pd.DataFrame(columns = ['words'])
    stop_words = set(stopwords.words('english'))
    #change to review.index after complete, this helps with runtime
    for i in df.index:
        word_tokens = word_tokenize(df['reviews.text'][i])
        word_tokens = [w.lower() for w in word_tokens]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                if w.isalpha():
                    filtered_sentence.append(w)
        words = words.append({'words': filtered_sentence}, ignore_index=True)
    df['words'] = words
    return df

def sentiment(df_clean):
    df_clean['reviews.rating'] = df_clean['reviews.rating'].replace([4, 5],'Positive')
    df_clean['reviews.rating'] = df_clean['reviews.rating'].replace(3,'Neutral')
    df_clean['reviews.rating'] = df_clean['reviews.rating'].replace([1, 2],'Negative')
    df_final = pd.DataFrame()
    df_final = df_final.append(df_clean[df_clean['reviews.rating']=='Positive'].head(1400)) 
    df_final = df_final.append(df_clean[df_clean['reviews.rating']=='Neutral'].head(1400))
    df_final = df_final.append(df_clean[df_clean['reviews.rating']=='Negative'].head(1400)).reset_index().drop(columns = ['index'])
    return df_final

def create_sentiment_train_test_set(df_final):
    split = StratifiedShuffleSplit(n_splits=3, test_size=0.25)
    for train_index, test_index in split.split(df_final,df_final['reviews.rating']): 
        strat_train = df_final.reindex(train_index)
        strat_test = df_final.reindex(test_index)
    
    X_train = strat_train['words'].reset_index().drop(columns = ['index']) 
    y_train = strat_train['reviews.rating'].reset_index().drop(columns = ['index'])
    X_test = strat_test['words'].reset_index().drop(columns = ['index'])
    y_test = strat_test['reviews.rating'].reset_index().drop(columns = ['index'])
    
    X_train_string = pd.DataFrame(columns = ['strings'])
    for words in X_train['words']:
        X_train_string = X_train_string.append({'strings': ' '.join(words)}, ignore_index=True)

    X_test_string = pd.DataFrame(columns = ['strings'])
    for words in X_test['words']:
        X_test_string = X_test_string.append({'strings': ' '.join(words)}, ignore_index=True)

    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(X_train_string['strings'])
    X_train_tf = vectorizer.transform(X_train_string['strings'])
    X_test_tf = vectorizer.transform(X_test_string['strings'])
    return y_train, y_test, X_train_tf, X_test_tf, vectorizer

def predict_LinearSVC(X_train_tf, X_test_tf, y_train, y_test):
    model = LinearSVC(random_state = 0)
    model.fit(X_train_tf,y_train)
    y_pred = model.predict(X_test_tf)  
    classifier_report('\nLinear State Vector Classifier', y_test, y_pred)

def predict_SVC(X_train_tf, X_test_tf, y_train, y_test):
    grid = SVC(C = 1, gamma = 1, random_state = 30)
    grid.fit(X_train_tf,y_train)
    grid_pred = grid.predict(X_test_tf)
    classifier_report('\nState Vector Classifier', y_test, grid_pred)
    return grid

def predict_RandomForestClassifier(X_train_tf, X_test_tf, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train_tf,y_train)
    y_pred = clf.predict(X_test_tf)  
    classifier_report('\nRandom Forest Classifier', y_test, y_pred) 

def predict_DecisionTreeClassifier(X_train_tf, X_test_tf, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train_tf,y_train)
    y_pred = clf.predict(X_test_tf)  
    classifier_report('\nDecision Tree Classifier', y_test, y_pred) 

def classifier_report(classifier, y_test, y_pred):
    print(classifier)
    print('\nPrediction Accuracy is ', math.ceil(metrics.accuracy_score(y_test, y_pred) * 100), '%')
    print(metrics.classification_report(y_test, y_pred))
    print('Confusion Matrix is ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

if __name__=='__main__':
    df = create_sentiment_dataframe()   # DataFrame with required features extracted from dataframe
    df_clean = stopwords_filter(df)     # DataFrame with stopwords filtered
    df_final = sentiment(df_clean)      # Classifying sentiments as positive/neutral/negative
    y_train, y_test, X_train_tf, X_test_tf, vectorizer1 = create_sentiment_train_test_set(df_final)
    predict_LinearSVC(X_train_tf, X_test_tf, y_train, y_test)
    model = predict_SVC(X_train_tf, X_test_tf, y_train, y_test)
    joblib.dump(model, 'sentiment.pkl')
    joblib.dump(vectorizer1, 'vectorizer1.pkl')

    svm = SVC(probability=True)
    svm.fit(X_train_tf, y_train)
    joblib.dump(svm, 'svm.pkl')
    predict_RandomForestClassifier(X_train_tf, X_test_tf, y_train, y_test)
    predict_DecisionTreeClassifier(X_train_tf, X_test_tf, y_train, y_test)