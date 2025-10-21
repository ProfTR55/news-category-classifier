import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate
import torch
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback


# GPU kontrolü
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Kullanılan cihaz:", device, torch.cuda.get_device_name(0) if device=="cuda" else "")


# ---------------------- #
# Label mapping
# ---------------------- #
label2id = {
    "güncel": 0, "genel": 1, "spor": 2, "siyaset": 3, "magazin": 4,
    "dünya": 5, "sağlık": 6, "yaşam": 7, "türkiye": 8, "planet": 9,
    "teknoloji": 10, "kültür-sanat": 11,"ekonomi": 12
}
id2label = {v: k for k, v in label2id.items()}


# ---------------------- #
# Veri Yükleme
# ---------------------- #
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df["label"] = train_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_test_split = train_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]


# ---------------------- #
#  Tokenizer ve Model
# ---------------------- #
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(device)


# ---------------------- #
# Tokenization Fonksiyonu
# ---------------------- #
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)


train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# ---------------------- #
# Metrik Tanımı
# ---------------------- #
metric_accuracy = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_accuracy.compute(predictions=preds, references=labels)
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "macro_f1": f1["f1"]}

# ---------------------- #
#  TrainingArguments
# ---------------------- #
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    evaluation_strategy= "epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)


# ---------------------- #
#  Trainer
# ---------------------- #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# ---------------------- #
#  Eğitim Başlatma
# ---------------------- #
trainer.train()


# ---------------------- #
#  Sonuçları Değerlendirme
# ---------------------- #
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)


# ---------------------- #
# Modeli Kaydetme
# ---------------------- #
model.save_pretrained("model/fine_tuned_bert")
tokenizer.save_pretrained("model/fine_tuned_bert")
print("Model ve tokenizer kaydedildi: model/fine_tuned_bert/")



