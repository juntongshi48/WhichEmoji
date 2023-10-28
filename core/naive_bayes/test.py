from pandas import *
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set() # use seaborn plotting style
from sklearn.metrics import accuracy_score, classification_report


train_file = read_csv("../dataset/data/processed/train.csv")
test_file = read_csv("../dataset/data/processed/test.csv")

train_file = train_file.replace('\t', ' ', regex=True)
test_file = test_file.replace('\t', ' ', regex=True)

# X_train, X_test, y_train, y_test
y_train = train_file.iloc[:, 0]
# X_train = train_file.iloc[:, 1].str.split('\t')
X_train = train_file.iloc[:, 1]
y_test = test_file.iloc[:, 0]
X_test = test_file.iloc[:, 1]
# X_test = test_file.iloc[:, 1].str.split('\t')


# column_1 = train_file.loc[:, ~train_file.isnull().all()].iloc[:, -2]

print("data[0]:")
print(X_train)

print("data[1]:")
print(y_train)


# # Model building
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# # Training the model with the training data
# model.fit(.data, train_data.target)
# # Predicting the test data categories
# predicted_categories = model.predict(test_data.data)

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_vect, y_train)

y_pred = nb.predict(X_test_vect)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))


