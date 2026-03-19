import pandas as pd

from sklearn.model_selection import train_test_split

from preprocess import clean_text
from resample_data import balance_dataset
from vectorize import vectorize_text
from train_models import train_models

from visualize import (
    plot_class_distribution,
    plot_balanced_distribution,
    plot_wordcloud,
    plot_model_comparison
)


def main():

    # Load dataset
    df = pd.read_csv("../data/hate_speech.csv")

    # Plot original dataset distribution
    plot_class_distribution(df)

    # Convert to binary classification
    df['class'] = df['class'].apply(lambda x: 1 if x == 2 else 0)

    # Clean tweets
    df['clean_tweet'] = df['tweet'].apply(clean_text)

    # Balance dataset
    df_balanced = balance_dataset(df)

    # Plot balanced dataset
    plot_balanced_distribution(df_balanced)

    # Generate word cloud of hate tweets
    hate_text = " ".join(df_balanced[df_balanced['class'] == 0]['clean_tweet'])
    plot_wordcloud(hate_text)

    # Convert text to TF-IDF
    X, vectorizer = vectorize_text(df_balanced['clean_tweet'])

    y = df_balanced['class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models and collect results
    results = train_models(X_train, X_test, y_train, y_test)

    # Plot model accuracy comparison
    plot_model_comparison(results)


if __name__ == "__main__":
    main()