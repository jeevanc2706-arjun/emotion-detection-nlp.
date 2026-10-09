import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("results", exist_ok=True)

# Small reproducible demonstration dataset.
# Replace this with the original project dataset when available.
data = pd.DataFrame({
    "text": [
        "I am extremely happy today",
        "This news made me joyful",
        "I feel wonderful and excited",
        "I am very sad about what happened",
        "This situation makes me unhappy",
        "I feel lonely and disappointed",
        "I am angry about this unfair decision",
        "This makes me furious",
        "I cannot believe how frustrating this is",
        "I am scared about what may happen",
        "This situation makes me nervous",
        "I feel worried and afraid",
        "I love this beautiful surprise",
        "I am delighted with the result",
        "I am upset about the mistake",
        "That decision made me angry",
        "I am worried about tomorrow",
        "I feel happy and positive",
        "I am disappointed by the outcome",
        "I am excited for the event",
        "This is frightening and stressful",
        "I am furious about the delay",
        "What a wonderful experience",
        "I feel unhappy today",
    ],
    "emotion": [
        "joy","joy","joy",
        "sadness","sadness","sadness",
        "anger","anger","anger",
        "fear","fear","fear",
        "joy","joy","sadness",
        "anger","fear","joy",
        "sadness","joy","fear",
        "anger","joy","sadness"
    ]
})

X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["emotion"],
    test_size=0.25,
    random_state=42,
    stratify=data["emotion"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SMOTE requires a dense matrix for this small demonstration.
X_train_balanced, y_train_balanced = SMOTE(
    random_state=42,
    k_neighbors=1
).fit_resample(
    X_train_tfidf.toarray(),
    y_train
)

model = SVC(kernel="linear", C=1.0)
model.fit(X_train_balanced, y_train_balanced)

y_pred = model.predict(X_test_tfidf.toarray())

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

labels = sorted(data["emotion"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Emotion Detection Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()

with open("results/model_output.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
