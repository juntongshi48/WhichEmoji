import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB


input_dir = "core/dataset/data/processed/train.csv"
data = pd.read_csv(input_dir, delimiter=',', header=None)
y = data[0]
X = data[1].str.split('\t').str.join(' ')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("done reading file")

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("start training")

# random forest
# clf = RandomForestClassifier()

# SVM
# clf = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3)  # 'hinge' gives a linear SVM

# multinomial naive bayes
# clf = MultinomialNB()

clf.fit(X_train_vec, y_train)

print("start prediction")

y_pred = clf.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")