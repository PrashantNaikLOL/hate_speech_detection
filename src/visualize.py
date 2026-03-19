import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


def plot_class_distribution(df):

    plt.figure(figsize=(6,4))
    sns.countplot(x='class', data=df)

    plt.title("Class Distribution Before Resampling")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig("../plots/class_distribution_before.png")
    plt.close()


def plot_balanced_distribution(df):

    plt.figure(figsize=(6,4))
    sns.countplot(x='class', data=df)

    plt.title("Class Distribution After Resampling")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.savefig("../plots/class_distribution_after.png")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"../plots/{model_name}_confusion_matrix.png")
    plt.close()


def plot_model_comparison(results):

    names = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(7,4))
    sns.barplot(x=names, y=scores)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")

    plt.savefig("../plots/model_comparison.png")
    plt.close()


def plot_wordcloud(text):

    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.title("Hate Speech Word Cloud")

    plt.savefig("../plots/hate_wordcloud.png")
    plt.close()