# Türkçe Haber Sınıflandırma Modeli (BERT)

Bu proje, [Hugging Face Transformers](https://huggingface.co/transformers/) kütüphanesi ve `dbmdz/bert-base-turkish-cased` modeli kullanılarak geliştirilmiş bir Türkçe haber metni sınıflandırma modelidir. Model, haber metinlerini verilen metinlere göre 13 farklı kategoriye ayırmak için fine-tune edilmiştir.

## 🚀 Özellikler

- **BERT Mimarisi**: Yüksek doğruluk için `dbmdz/bert-base-turkish-cased` modeli temel alınmıştır.
- **Fine-Tuning**: Özel bir veri seti üzerinde modelin yeniden eğitilmesi.
- **Tahmin Scripti**: Eğitilmiş model ile interaktif olarak yeni haber metinlerini sınıflandırma imkanı.
- **Değerlendirme**: Eğitim sırasında `accuracy` ve `macro_f1` metrikleri ile model performansının ölçümü.

## 📚 Kategoriler

Model, haberleri aşağıdaki 13 kategoriden birine sınıflandırır:
- güncel
- genel
- spor
- siyaset
- magazin
- dünya
- sağlık
- yaşam
- türkiye
- planet
- teknoloji
- kültür-sanat
- ekonomi

## 📦 Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1.  **Projeyi Klonlayın:**
    ```bash
    git clone <repository-url>
    cd <repository-adınız>
    ```

2.  **Sanal Ortam Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    ```
    - Windows için:
      ```bash
      venv\Scripts\activate
      ```
    - macOS/Linux için:
      ```bash
      source venv/bin/activate
      ```

3.  **Bağımlılıkları Yükleyin:**
    Proje için gerekli olan kütüphaneleri `requirements.txt` dosyasını kullanarak yükleyin.
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Kullanım

### 1. Veri Hazırlığı

- Eğitim için `data/train.csv` ve `data/test.csv` dosyalarını kendi veri setinizle değiştirebilirsiniz.
- CSV dosyaları `text` ve `category` sütunlarını içermelidir.

### 2. Modeli Eğitme

Modeli yeniden eğitmek için aşağıdaki komutu çalıştırın. Bu script, `data` klasöründeki verileri kullanarak modeli eğitecek ve en iyi modeli `model/fine_tuned_bert` klasörüne kaydedecektir.

```bash
python train_bert_finetune.py
```

### 3. Tahmin Yapma

Eğitilmiş modeli kullanarak interaktif bir şekilde haber başlıklarını veya metinlerini sınıflandırmak için `predict_news.py` scriptini çalıştırın.

```bash
python predict_news.py
```
Script başladığında, sizden bir haber başlığı girmeniz istenecektir. Model tahminini yapacak ve sonucu ekrana yazdıracaktır. Çıkmak için `q` yazabilirsiniz.

## 📂 Proje Yapısı

```
.
├── data/
│   ├── train.csv
│   └── test.csv
├── model/
│   └── fine_tuned_bert/      # Eğitilmiş modelin kaydedildiği yer
├── results/                  # Eğitim sırasında oluşturulan checkpoint'ler
├── train_bert_finetune.py    # Model eğitim scripti
├── predict_news.py           # Model ile tahmin yapma scripti
├── visualize_model.py        # Model sonuçlarını görselleştirme scripti
├── requirements.txt          # Gerekli kütüphaneler
└── README.md                 # Proje açıklaması
```
