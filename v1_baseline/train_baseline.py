import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib



train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

x_train = train_df["text"]
y_train = train_df["category"]

x_test = test_df["text"]
y_test = test_df["category"]

# Burda kelimeleri vectorlere ayırıyoruz.
vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1,3), min_df=3, max_df=0.9)

#Verileri sigdirdik
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

#Logistic Regression modelinden geçiriyoruz vectorleri
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(x_train, y_train)

#Performans olcumu
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

accuracy = clf.score(x_test, y_test)
print("Accuracy:", accuracy)


joblib.dump(clf, "model/baseline_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
