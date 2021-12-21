import math
import joblib
import warnings
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from predict_sentiment import stopwords_filter
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

def create_aspect_dataframe():
    data = pd.read_csv('dataset.csv',low_memory=False,encoding='ISO-8859-1')
    data = data[['primaryCategories','reviews.text']]
    data.drop(data[(data['primaryCategories'] =='Office Supplies')].index, inplace=True)
    data.drop(data[(data['primaryCategories'] =='Animals & Pet Supplies')].index, inplace=True)
    data.drop(data[(data['primaryCategories'] =='Home & Garden')].index, inplace=True)
    data.drop(data[(data['primaryCategories'] =='Electronics,Furniture')].index, inplace=True)
    return data

def aspect(df_clean):
    le = LabelEncoder()
    df_clean['primaryCategories'] = le.fit_transform(df_clean['primaryCategories'])
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(0,'Electronics')
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(1,'Electronics & Hardware')
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(2,'Electronics & Media')
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(3,'Health & Beauty')
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(4,'Office Supplies & Electronics')
    df_clean['primaryCategories'] = df_clean['primaryCategories'].replace(5,'Toys, Games & Electronics')
    df_final = pd.DataFrame()
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Electronics'].head(209))
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Electronics & Hardware'].head(209))
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Electronics & Media'].head(209))
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Health & Beauty'].head(209))
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Office Supplies & Electronics'].head(209))
    df_final = df_final.append(df_clean[df_clean['primaryCategories']=='Toys, Games & Electronics'].head(209)).reset_index().drop(columns = ['index'])
    return df_final

def create_aspect_train_test_set(df_final):
    split = StratifiedShuffleSplit(n_splits=6, test_size=0.25)
    for train_index, test_index in split.split(df_final,df_final['primaryCategories']): 
        strat_train = df_final.reindex(train_index)
        strat_test = df_final.reindex(test_index)
    
    X_train = strat_train['words'].reset_index().drop(columns = ['index']) 
    y_train = strat_train['primaryCategories'].reset_index().drop(columns = ['index'])
    X_test = strat_test['words'].reset_index().drop(columns = ['index'])
    y_test = strat_test['primaryCategories'].reset_index().drop(columns = ['index'])
    
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
    classifier_report('LinearSVC', y_test, y_pred)
    return model

def predict_SVC(X_train_tf, X_test_tf, y_train, y_test):
    grid = SVC(C = 1, gamma = 1, random_state = 30)
    grid.fit(X_train_tf,y_train)
    grid_pred = grid.predict(X_test_tf)
    classifier_report("GridSVC", y_test, grid_pred)

def predict_RandomForestClassifier(X_train_tf, X_test_tf, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train_tf,y_train)
    y_pred = clf.predict(X_test_tf)  
    classifier_report("Random Forest Classifier", y_test, y_pred) 

def predict_DecisionTreeClassifier(X_train_tf, X_test_tf, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train_tf,y_train)
    y_pred = clf.predict(X_test_tf)  
    classifier_report("Decision Tree Classifier", y_test, y_pred) 

def classifier_report(classifier,y_test,y_pred):
    print(classifier)
    print(']\nPrediction Accuracy is ', math.ceil(metrics.accuracy_score(y_test, y_pred) * 100), '%')
    print(metrics.classification_report(y_test, y_pred))
    print('Confusion Matrix is ', metrics.confusion_matrix(y_test, y_pred), sep='\n')

if __name__=='__main__':
    df = create_aspect_dataframe()   # DataFrame with required features extracted from dataframe
    df_clean = stopwords_filter(df)     # DataFrame with stopwords filtered
    df_final = aspect(df_clean)      # Classifying sentiments as positive/neutral/negative
    y_train, y_test, X_train_tf, X_test_tf, vectorizer2 = create_aspect_train_test_set(df_final)
    model = predict_LinearSVC(X_train_tf, X_test_tf, y_train, y_test)
    predict_SVC(X_train_tf, X_test_tf, y_train, y_test)
    joblib.dump(model, 'aspect.pkl')
    joblib.dump(vectorizer2, 'vectorizer2.pkl')
    predict_RandomForestClassifier(X_train_tf, X_test_tf, y_train, y_test)
    predict_DecisionTreeClassifier(X_train_tf, X_test_tf, y_train, y_test)