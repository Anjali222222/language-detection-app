import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# dataset load
df = pd.read_csv("Language Detection.csv")

# features & labels
X = df["Text"]
y = df["Language"]

# vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# model train
model = MultinomialNB()
model.fit(X, y)

# 🔥 SAVE FILES (MOST IMPORTANT)
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Model saved successfully ✅")