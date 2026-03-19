from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score


def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Naive Bayes": MultinomialNB(),
        "SGD": SGDClassifier(random_state=42),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        print("\n==============================")
        print(name)
        print("==============================")

        print("Accuracy:", accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))