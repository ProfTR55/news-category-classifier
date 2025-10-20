import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split



def clean_text(text):

    if not isinstance(text, str):
        return ""

    # Küçük harfe çevir
    text = text.lower()

    # Linkleri kaldır
    text = re.sub(r"http\S+|www\S+", "", text)

    # Noktalama ve özel karakterleri kaldır
    text = re.sub(r"[^a-zçğıöşü\s]", " ", text)

    # Fazla boşlukları kaldır
    text = re.sub(r"\s+", " ", text).strip()

    # Türkçe stopword'leri kaldır
    stop_words = set(stopwords.words("turkish"))
    words = [w for w in text.split() if w not in stop_words]

    return " ".join(words)

def preprocess_dataset(input_path, output_path):

    print("Veri okunuyor...")
    df = pd.read_excel(input_path)

    # Gerekli sütunlar var mı kontrol et
    expected_cols = {"headline", "content", "category"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Beklenen sütunlar bulunamadı! Gerekli sütunlar: {expected_cols}")

    print("Başlık ve içerik birleştiriliyor...")
    df["headline"] = df["headline"].fillna("")
    df["content"] = df["content"].fillna("")
    df["text"] = df["headline"] + " " + df["content"]

    print("Metinler temizleniyor...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Gereksiz sütunları at
    df = df[["clean_text", "category"]].rename(columns={"clean_text": "text"})

    print("Train/test ayrımı yapılıyor...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category"])

    # Dosyaları kaydet
    print("Kaydediliyor...")
    train_df.to_csv(f"{output_path}/train.csv", index=False)
    test_df.to_csv(f"{output_path}/test.csv", index=False)

    print("İşlem tamamlandı! -> train.csv ve test.csv hazır.")

if __name__ == "__main__":
    # Örnek kullanım
    preprocess_dataset(
        input_path="C:/Users/Doğukan/Desktop/news/data/news.xls",   # veri setinin yolu
        output_path="data"                 # çıktıların kaydedileceği klasör
    )
