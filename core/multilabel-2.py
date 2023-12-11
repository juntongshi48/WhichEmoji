import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import random
from sklearn.preprocessing import StandardScaler

# Load training and test data
train_df = pd.read_csv('core/dataset/data/multilabel/train.csv')
test_df = pd.read_csv('core/dataset/data/multilabel/test.csv')

# Assuming the first column is labels and the second column is tweets
X_train = train_df.iloc[:, 1]
y_train = train_df.iloc[:, 0]
X_test = test_df.iloc[:, 1]
y_test = test_df.iloc[:, 0]

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse data
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)

# Train the model
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))  # Increase max_iter
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='macro')
print(f'Accuracy: {accuracy}, F1 Score: {f1}')

num_samples = 5
sample_indices = random.sample(range(len(X_test)), num_samples)
sample_texts = X_test.iloc[sample_indices]
sample_labels = y_test.iloc[sample_indices]

# Vectorize the sample texts
sample_texts_tfidf = vectorizer.transform(sample_texts)

# Make predictions on the samples
sample_predictions = model.predict(sample_texts_tfidf)

# Display actual and predicted labels for the samples
for text, actual, predicted in zip(sample_texts, sample_labels, sample_predictions):
    print("Tweet:", text)
    print("Actual Label:", actual)
    print("Predicted Label:", predicted)
    print("---------")
