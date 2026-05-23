from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import load_data
from preprocess import clean_text
import pandas as pd
import joblib
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def train():

    data_path = os.path.join(base_dir, "data", "spam.csv")
    df = load_data(data_path)

    # Load and merge custom anonymized email spam data
    new_data_path = os.path.join(base_dir, "data", "anonymized_spam.csv")
    if os.path.exists(new_data_path):
        df_new = load_data(new_data_path)
        df_new = df_new.dropna(subset=['text'])
        df = pd.concat([df, df_new], ignore_index=True)
        print(f"Merged {len(df_new)} custom anonymized email spam samples. Total dataset size: {len(df)}")

    # Clean text
    df = df.dropna(subset=['text'])
    df['text'] = df['text'].apply(clean_text)

    # Features and labels
    X = df['text']
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =====================================
    # NAIVE BAYES MODEL
    # =====================================

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', MultinomialNB())
    ])

    nb_pipeline.fit(X_train, y_train)

    nb_predictions = nb_pipeline.predict(X_test)

    nb_accuracy = accuracy_score(y_test, nb_predictions)

    print(f"\nNaive Bayes Accuracy: {nb_accuracy:.4f}")

    # =====================================
    # LOGISTIC REGRESSION MODEL
    # =====================================

    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', LogisticRegression())
    ])

    lr_pipeline.fit(X_train, y_train)

    lr_predictions = lr_pipeline.predict(X_test)

    lr_accuracy = accuracy_score(y_test,lr_predictions)

    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

    # =====================================
    # SAVE BEST MODEL
    # =====================================

    if lr_accuracy > nb_accuracy:

        best_model = lr_pipeline

        print("\nBest Model: Logistic Regression")

    else:

        best_model = nb_pipeline

        print("\nBest Model: Naive Bayes")

    model_path = os.path.join(base_dir, "models", "spam_pipeline.pkl")
    joblib.dump(best_model, model_path)

    print("\nModel saved successfully.")


if __name__ == "__main__":

    train()