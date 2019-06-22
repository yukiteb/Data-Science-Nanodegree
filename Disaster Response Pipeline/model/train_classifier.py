# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, classification_report


def load_data(database_filepath):
    """Load data
    Args:
        database_filepath (string): File path of database
    Returns:
        X: Messages
        Y: Labeled categories
        category_names: Name of categories
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('ResponseCategoryTable',engine.connect())
    #Drop related-2
    df = df[df.related !=2]
    #Drop original message and NaN
    df.drop(['original'], axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)

    X = df['message']
    Y = df.drop(['id','message','genre'], axis=1)
    category_names = Y.columns
    print (category_names)
    return X, Y, category_names

def tokenize(text):
    """Tokenize a text
    Args:
        text (string): text to be tokenized
    Returns:
        Tokenized text as a list
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build a model
    Args:
        None
    Returns:
        modle (object) : model pipeline
    """

    pipeline = Pipeline([('vect', TfidfVectorizer(tokenizer=tokenize)),
                     ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=123))))
                      ])

    parameters = {'clf__estimator__estimator__C': [1, 10, 50]}

    model = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model performance
    Args:
        model (object): model to be evaluated
        X_test:
        Y_test:
        category_names:
    Returns:
        None
    """
    y_pred = model.predict(X_test.tolist())
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Save a model
    Args:
        model (object): model to be saved
        model_filepath (string): file path of the model to be saved
    Returns:
        None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

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
