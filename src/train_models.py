from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score

from visualize import plot_confusion_matrix


def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Naive Bayes": MultinomialNB(),
        "SGD": SGDClassifier(random_state=42),
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        results[name] = acc

        print("\n==============================")
        print(name)
        print("==============================")
        print("Accuracy:", acc)
        print(classification_report(y_test, predictions))

        plot_confusion_matrix(y_test, predictions, name)

    return results