#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def main():
    # Read the data
    human_data = pd.read_table('/Users/montaha.chowdhury/Documents/Comp Bio Learning/ML_DNA Sequencing/Datasets/human.txt')
    chimp_data = pd.read_table('/Users/montaha.chowdhury/Documents/Comp Bio Learning/ML_DNA Sequencing/Datasets/chimpanzee.txt')
    dog_data = pd.read_table('/Users/montaha.chowdhury/Documents/Comp Bio Learning/ML_DNA Sequencing/Datasets/dog.txt')

    # Process the data
    human_data['words'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)
    chimp_data['words'] = chimp_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    chimp_data = chimp_data.drop('sequence', axis=1)
    dog_data['words'] = dog_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    dog_data = dog_data.drop('sequence', axis=1)

    # Convert k-mers to string sentences
    human_texts = [' '.join(words) for words in human_data['words']]
    y_data = human_data.iloc[:, 0].values

    # Bag of words
    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)

    # Train the model
    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("Confusion matrix \n")
    print(pd.crosstab(pd.Series(y_test, name="Actual"), pd.Series(y_pred, name="Predicted")))
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average="weighted")
    recall = recall_score(y_test, y_predicted, average="weighted")
    f1 = f1_score(y_test, y_predicted, average="weighted")
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    main()