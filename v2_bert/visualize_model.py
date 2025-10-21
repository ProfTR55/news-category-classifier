import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["WANDB_DISABLED"] = "true"

# 1️ Modeli yükle
model_path = "model/fine_tuned_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 2️ Test verisini yükle
label2id = {
    "güncel": 0, "genel": 1, "spor": 2, "siyaset": 3, "magazin": 4,
    "dünya": 5, "sağlık": 6, "yaşam": 7, "türkiye": 8, "planet": 9,
    "teknoloji": 10, "kültür-sanat": 11, "ekonomi": 12
}
test_df = pd.read_csv("data/test.csv")
test_df["label"] = test_df["category"].map(label2id)
test_dataset = Dataset.from_pandas(test_df)

# 3️ Tokenize et
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 4️ Trainer sadece değerlendirme için
trainer = Trainer(model=model, tokenizer=tokenizer)
predictions = trainer.predict(test_dataset)

# 5️ Sonuçları çıkar
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix - Fine-Tuned BERT")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("results/confusion_matrix.png")
plt.show()

report = classification_report(y_true, y_pred, target_names=list(label2id.keys()))
with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("✅ Confusion matrix and report saved!")
