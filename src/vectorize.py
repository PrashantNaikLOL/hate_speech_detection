from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(data):

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(data)

    return X, vectorizer