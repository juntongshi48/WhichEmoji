import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from main import plot_confusion_matrix

import pdb

id2label = {0: "enraged_face", 
             1: "face_holding_back_tears", 
             2:"face_savoring_food", 
             3: "face_with_tears_of_joy", 
             4: "fearful_face", 
             5: "hot_face", 
             6: "sun", 
             7: "loudly_crying_face", 
             8: "smiling_face_with_sunglasses", 
             9: "thinking_face"}

train_dir = "core/dataset/data/processed/train.csv"
test_dir = "core/dataset/data/processed/test.csv"
train_data = pd.read_csv(train_dir, delimiter=',', header=None)
X_train = train_data[1].str.split('\t').str.join(' ')
y_train = train_data[0]

test_data = pd.read_csv(test_dir, delimiter=',', header=None)
X_test = test_data[1].str.split('\t').str.join(' ')
y_test = test_data[0]

print("done reading file")

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("start training")

model_name = 'SVM'

# random forest
# clf = RandomForestClassifier()

if model_name == 'SVM':
    # SVM
    clf = SGDClassifier(loss='squared_hinge', max_iter=10000, tol=1e-4)  # 'hinge' gives a linear SVM
elif model_name == 'NB':
    # multinomial naive bayes
    clf = MultinomialNB()
else:
    raise ValueError("Model with the input name hasn't been implemented")

clf.fit(X_train_vec, y_train)

print("start prediction")

y_pred_test = clf.predict(X_test_vec)
y_pred_train = clf.predict(X_train_vec)


accuracy = accuracy_score(y_train, y_pred_train)
print(f"Train Accuracy: {accuracy*100:.4f}%")

accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {accuracy*100:.4f}%")

f1_macro = f1_score(y_test, y_pred_test, average='macro')
print(f"Test F1 Macro: {f1_macro}")

confus_mtx = confusion_matrix(y_test, y_pred_test, normalize='true')
plot_confusion_matrix(id2label.values(), confus_mtx, f'Confusion Matrix of {model_name}', f'CM_{model_name}.png')