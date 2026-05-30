# Perfume Recommendation System
### Sentiment & Numerical Feature-Based Recommendations using BERT, RoBERTa, and DistilBERT

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning system that recommends perfumes by combining **Aspect-Based Sentiment Analysis (ABSA)** on user reviews with structured numerical features (longevity, sillage, projection, value). Built as a Master's thesis project at Bina Nusantara University.

---

## The Problem

Most existing perfume recommendation systems match only on scent notes (e.g. "woody", "citrus"). This ignores subjective user experiences — how long a fragrance lasts, its projection, its perceived quality and value. This project closes that gap by incorporating sentiment from real user reviews alongside traditional metadata.

---

## How It Works

1. **Data Collection** — ~160,000 user reviews across 740 unique perfumes scraped from [Fragrantica](https://www.fragrantica.com/)
2. **Preprocessing** — Text cleaning (lowercase, remove HTML/emojis/punctuation), lemmatization via NLTK, stopword removal. Numerical attributes (longevity, sillage, projection, value) normalized via weighted scoring from user votes
3. **Aspect-Based Sentiment Analysis (ABSA)** — Each review is broken into aspect-sentiment pairs across 6 aspects: `longevity`, `sillage`, `versatility`, `value`, `scent_quality`, `overall`. Three Transformer models fine-tuned for classification: positive / neutral / negative
4. **Recommendation Engine** — Cosine similarity across a hybrid feature vector combining:
   - Sentiment embeddings from ABSA
   - Perfume note metadata (TF-IDF)
   - Numerical vote scores (longevity, sillage, projection, value)
   - Gender classification
5. **Output** — Top-K ranked perfume recommendations per input perfume

---

## Models Compared

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---|---|---|---|
| **BERT** | **96.91%** | **96.20%** | **96.22%** | **96.18%** |
| RoBERTa | 96.15% | 95.42% | 95.76% | 95.09% |
| DistilBERT | 96.08% | 95.43% | 95.72% | 95.15% |

**DistilBERT was selected as the deployment model.** Although BERT achieved the best classification metrics, DistilBERT produced the highest average similarity scores in the recommendation module (0.84–0.86 vs BERT's 0.84) while being 40% smaller and faster — only a 1% margin in weighted F1.

---

## Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| Batch Size | 16 |
| Epochs | 5 |
| Max Token Length | 256 |
| Loss Function | Cross-Entropy |
| Framework | PyTorch + HuggingFace Transformers |

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/bert_model.ipynb` | Full pipeline — data preprocessing, ABSA training (all 3 models compared), and recommendation system |
| `notebooks/distilbert_model.ipynb` | DistilBERT-only training run |
| `notebooks/roberta_model.ipynb` | RoBERTa-only training run |

> **Note:** The notebooks were developed and run on Kaggle. The dataset path references `/kaggle/input/fragrantica-perfumes/` — update this if running locally.

---

## Dataset

- **Source:** [Fragrantica](https://www.fragrantica.com/) (via Kaggle dataset) [Dataset Used](https://www.kaggle.com/datasets/wilsonanthony/fragrantica-perfumes)
- **Size:** ~160,000 reviews across 740 unique perfumes
- **Features:** Perfume name, brand, release year, gender classification, user vote data (longevity, sillage, projection, value), raw review text
- **Splits:** 5 Excel files merged and aggregated by perfume ID

---

## Key Results

- **BERT** had the lowest misclassification rate — 0 positive reviews mislabeled as negative, 9 negatives mislabeled as positive out of ~11,800 test samples
- **DistilBERT** produced the highest recommendation similarity scores across all 3 test cases (Terre d'Hermès, Coco Mademoiselle, CK One)
- The system successfully demonstrates **sentiment-awareness**: flanker perfumes (e.g. Terre d'Hermès Eau Intense Vetiver) ranked first for matching input; high-quality niche fragrances ranked above average ones; gender-flexible recommendations produced for unisex inputs

---

## Requirements

```
transformers
torch
scikit-learn
pandas
numpy
nltk
openpyxl
tqdm
scipy
seaborn
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Paper

This project was published as:

> **Wilson Anthony Widjaja**, Maria Susan Anggreainy. *"Perfume Recommendation System Based on Sentiment and Numerical Features: A Comparative Study of BERT, RoBERTa, and DistilBERT."* Bina Nusantara University, 2025.

---

## Author

**Wilson Anthony Widjaja**  
Master of Computer Science — Bina Nusantara University  
[LinkedIn](https://www.linkedin.com/in/wilson-anthony-widjaja)
