# Turkish News Classification Project

This project provides two different models for classifying Turkish news articles into various categories such as sports, politics, economy, and more. It includes a classic machine learning baseline and a modern deep learning approach using a Transformer-based model.

## Models

This repository contains two distinct models, each in its own directory.

### 1. `v1_baseline`: TF-IDF + Logistic Regression

This model serves as a simple and fast baseline for the text classification task.

-   **Architecture**: It uses a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorizer to convert text into numerical features and a **Logistic Regression** classifier to perform the classification.
-   **Technology**: Built with `scikit-learn`, `pandas`, and `nltk`.
-   **Performance**: This model provides a solid starting point. It performs very well for distinct categories like "spor" (sports) but struggles with more nuanced or overlapping categories.
-   **Details**: You can find the training script, saved model, and dependencies in the `v1_baseline/` directory.

### 2. `v2_bert`: Fine-Tuned BERT Model

This is a more advanced model that leverages a pre-trained BERT model for higher accuracy and better semantic understanding.

-   **Architecture**: It uses the `dbmdz/bert-base-turkish-cased` model, a BERT model pre-trained on a large corpus of Turkish text. The model is then **fine-tuned** on the specific news classification dataset.
-   **Technology**: Built with `Hugging Face Transformers`, `PyTorch`, and `datasets`.
-   **Performance**: This model is expected to significantly outperform the baseline, as it can understand the context and semantics of the news text more effectively.
-   **Details**: The `v2_bert/` directory contains everything needed to train the model, run predictions, and manage dependencies.

## Project Structure

```
.
├── v1_baseline/
│   ├── train_baseline.py   # Script to train the TF-IDF + Logistic Regression model
│   ├── model/              # Saved baseline model
│   └── requirements.txt    # Dependencies for the baseline model
│
└── v2_bert/
    ├── train_bert_finetune.py  # Script to fine-tune the BERT model
    ├── predict_news.py         # Script to classify news with the BERT model
    ├── model/                  # Saved fine-tuned BERT model
    └── requirements.txt        # Dependencies for the BERT model
```

## How to Use

First, clone the repository to your local machine.

### Running the Baseline Model (v1)

1.  **Navigate to the baseline directory:**
    ```bash
    cd v1_baseline
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    ```bash
    python train_baseline.py
    ```
    This will train the model and save the `baseline_model.pkl` and `vectorizer.pkl` inside the `v1_baseline/model/` directory.

### Running the BERT Model (v2)

1.  **Navigate to the BERT directory:**
    ```bash
    cd v2_bert
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the training script:**
    ```bash
    python train_bert_finetune.py
    ```
    This will fine-tune the BERT model and save the best version to the `v2_bert/model/fine_tuned_bert/` directory.

4.  **Make predictions:**
    To classify new text, run the interactive prediction script:
    ```bash
    python predict_news.py
    ```
