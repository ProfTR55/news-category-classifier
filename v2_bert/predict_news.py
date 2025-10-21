import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modeli yükle
model_path = "model/fine_tuned_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = {
    0: "güncel", 1: "genel", 2: "spor", 3: "siyaset", 4: "magazin",
    5: "dünya", 6: "sağlık", 7: "yaşam", 8: "türkiye", 9: "planet",
    10: "teknoloji", 11: "kültür-sanat", 12: "ekonomi"
}

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label[pred]

print("📊 Model yüklendi. Çıkmak için 'q' yazın.\n")
while True:
    text = input("📰 Haber başlığını girin: ")
    if text.lower() == "q":
        break
    print("🔮 Tahmin:", predict_text(text), "\n")
