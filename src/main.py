import pandas as pd

from sklearn.model_selection import train_test_split

from preprocess import clean_text
from resample_data import balance_dataset
from vectorize import vectorize_text
from train_models import train_models


def main():

    df = pd.read_csv("../data/hate_speech.csv")

    # Convert to binary classification
    df['class'] = df['class'].apply(lambda x: 1 if x == 2 else 0)

    # Clean text
    df['clean_tweet'] = df['tweet'].apply(clean_text)

    # Balance dataset
    df_balanced = balance_dataset(df)

    X, vectorizer = vectorize_text(df_balanced['clean_tweet'])

    y = df_balanced['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()