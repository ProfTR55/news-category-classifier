# 📰 Turkish News Category Classifier

A baseline **Turkish news classification model** built using **TF-IDF** features and **Logistic Regression**.  
This project aims to classify news articles into multiple categories such as *spor, ekonomi, magazin, siyaset, teknoloji*, and more.

---

## 🚀 Project Overview

This project demonstrates the end-to-end process of **text classification in Turkish**:

1. **Data Preprocessing**  
   - Combined `headline` and `content` fields  
   - Cleaned punctuation, links, and stopwords  
   - Split data into `train.csv` and `test.csv`  

2. **Feature Extraction (TF-IDF)**  
   - Used up to **10,000 features**  
   - Included **1-3 word n-grams**  
   - Filtered rare & common terms (`min_df=3`, `max_df=0.9`)  

3. **Model Training**  
   - **Logistic Regression** with `class_weight='balanced'`  
   - Increased `max_iter` to ensure convergence  

4. **Evaluation**  
   - Evaluated using `accuracy`, `F1-score`, and `confusion matrix`

---

## 📊 Results (v1.0 – Baseline Model)

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.57 |
| **Macro F1** | 0.51 |
| **Weighted F1** | 0.55 |

### 🔹 Class-wise Performance Highlights
| Category | F1-score | Comment |
|-----------|-----------|----------|
| **spor** | 0.91 | Excellent separation |
| **magazin** | 0.67 | Strong |
| **sağlık** | 0.65 | Good generalization |
| **teknoloji** | 0.62 | Clear improvement after tuning |
| **genel / yaşam** | <0.20 | Semantically overlapping, needs embedding-based model |

---

## 🧠 Confusion Matrix Example
Below is the confusion matrix visualization generated with Seaborn:

```python
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.show()

