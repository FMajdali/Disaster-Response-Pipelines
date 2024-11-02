import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from sqlalchemy import create_engine
import joblib
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit

#This class inherits from the sklearn BaseEstimator
#It is an estimator which returns a feature of messgae length 
class MessageLength(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[len(text)] for text in X])

#This class inherits from the sklearn BaseEstimator
#It is an estimator which returns a feature of words count in the message
class MessageWordsCount(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[len(text.split())] for text in X])

#This class inherits from the sklearn BaseEstimator
#It is an estimator which returns a feature of 1 if the
#words ["water","food","earthquake"] are in the message, else it returns 0
class WordInMessage(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lst = []
        common_words = ["water","food","earthquake"]
        for word in common_words:
            lst.append(np.array([[1] if word in text.lower() else [0] for text in X]).reshape(-1,1))
        return np.concatenate(lst,axis=1)

#This function loads the messages and catgories data from SQL Database file
#It outputs The independet varibale which in our case is the messages,
#the labeles which in our case is the catgories, and lables names
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages',con= engine)
    cols = ['related', 'request', 'offer',
       'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
       'security', 'military', 'water', 'food', 'shelter',
       'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
       'infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
       'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather', 'direct_report','not_relevent']
    X = df["message"].copy()
    Y = df[cols].copy()
    
    return X, Y, cols

#This function takes a string as an input then it tokenize it and lemmatize it
# the out put is a list  of clean tokens
def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = nltk.corpus.stopwords.words("english")
    urls_in_text = re.findall(url_regex,text)
    
    for url in urls_in_text:
        text = text.replace(url,"url")
    
    tokens = word_tokenize(text)
    clean_tokens = []
    
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
    #remove stop words
    clean_tokens = [t for t in clean_tokens if t not in stop_words]
    clean_tokens = [t for t in clean_tokens if t.isalpha()]
    return clean_tokens

#This function builds a sklearn pipline where the estimator is a random forest
#The output of this function is a pipeline
def build_model():
    best_params = {'clf__estimator__criterion': 'gini',
                    'clf__estimator__max_depth': 5,
                    'clf__estimator__min_samples_leaf': 10,
                     'clf__estimator__min_samples_split': 2}
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_features', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
        ])),
            ('length_features', MessageLength()),
                ('words_count',MessageWordsCount()),
                ('word',WordInMessage())
    ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100),n_jobs=-1))
    ])
    
    pipeline.set_params(**best_params)
    
    return pipeline

#This function evalute a trained model in terms of average precision, average recall, and averge f1 score
#It prints the results
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    df_results = pd.DataFrame(columns = ["col_name","avg_precision","avg_recall","avg_f1_score","support"])
    df_pred = pd.DataFrame(Y_pred, columns= category_names)
    for col in category_names:
        res = classification_report(Y_test[col],df_pred[col])
        res = res.split("avg / total")[1][:-1].split("      ")
        res[0] = col
        df_results.loc[len(df_results)] = res
    print(df_results)

#This function saves the input model as a pkl file 
def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
