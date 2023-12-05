import os

import sklearn.feature_extraction.text as sk_text
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sys import argv

from typing import Union

import csv


def main(args):
    # load data
    vectorizer = sk_text.CountVectorizer()
    documents = []

    dir_name = args[1]
    for file in os.listdir(dir_name):
        if not file.endswith(".txt"):
            continue
        try:
            documents.append(open(dir_name + "/" + file, "r", encoding="utf-8").read())
        except Exception:
            documents.append(open(dir_name + "/" + file, "r", encoding="utf-16").read())
    vectorizer.fit(raw_documents=documents)

    # train
    model = LogisticRegression(max_iter=10000, C=1/32)
    with open(f"{dir_name}_data_collapsed.csv", "r") as f:
        train(model, vectorizer, csv.reader(f))

    # test
    performance = test(model, vectorizer)

    # evaluate
    evaluate(performance)

    # save model
    # save_model()

    # same for nb
    model = GaussianNB()
    with open(f"{dir_name}_data_collapsed.csv", "r") as f:
        train(model, vectorizer, csv.reader(f))
    performance = test(model, vectorizer)
    evaluate(performance)


def train(model: Union[LogisticRegression, GaussianNB], vectorizer: sk_text.CountVectorizer, data: csv.reader):
    # construct dataset
    fit = model.fit if isinstance(model, LogisticRegression) else model.partial_fit
    sentences = []
    features = []
    true_labels = []
    for sentence, *labels in data:
        sentences.append(sentence)
        features.append(vectorizer.transform([sentence]).toarray()[0])
        true_labels.append(1 if any([label == '1' for label in labels]) else 0)
    fit(features, true_labels, [0, 1] if isinstance(model, GaussianNB) else None)


def test(model: Union[LogisticRegression, GaussianNB], vectorizer: sk_text.CountVectorizer) -> list[str]:
    # test on test set
    samples = []
    true_labels = []
    with open("test_data_collapsed.csv", "r") as f:
        for sentence, *labels in csv.reader(f):
            features = vectorizer.transform([sentence]).toarray()[0]
            samples.append(features)
            true_labels.append(1 if any([label == '1' for label in labels]) else 0)
    samples = model.predict(samples)
    # label TP, TN, FP, FN
    performance = []
    for sample, label in list(zip(samples, true_labels)):
        if sample == 1 and label == 1:
            performance.append("TP")
        elif sample == 0 and label == 0:
            performance.append("TN")
        elif sample == 1 and label == 0:
            performance.append("FP")
        elif sample == 0 and label == 1:
            performance.append("FN")
        else:
            raise Exception("Something went wrong.")
    return performance


def evaluate(performance_vector: list[str]):
    # calculate accuracy, precision, recall, F1
    accuracy = (performance_vector.count("TP") + performance_vector.count("TN")) / len(performance_vector)
    precision = performance_vector.count("TP") / (performance_vector.count("TP") + performance_vector.count("FP"))
    recall = performance_vector.count("TP") / (performance_vector.count("TP") + performance_vector.count("FN"))
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}")


def save_model():
    pass


if __name__ == "__main__":
    main(argv)
