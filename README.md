# Analyzing Inherent and Learned Bias in Sentiment Models

### *Logistic Regression • VADER • BERT (IMDb, SST-2, Jigsaw, EEC)*

This repository contains all code, trained models, and analysis notebooks for the paper **“Analyzing Inherent and Learned Bias in Sentiment Models: A Comparative Study of Logistic Regression, VADER, and BERT.”**
The project evaluates both **accuracy** and **bias** across three sentiment systems using standardized datasets and fairness metrics.

---

## 1. Overview

Modern sentiment analysis models often exhibit **bias** due to linguistic priors, data imbalance, or contextual encoding.
This repository compares:

* **Logistic Regression (TF–IDF baseline)**
* **VADER lexicon-based sentiment model**
* **Fine-tuned BERT variants (SST-2 baseline & Jigsaw-biased model)**

We use the **Equity Evaluation Corpus (EEC)** to quantify bias through:

* **Average Sentiment Difference (ASD)**
* **95th Percentile Absolute Difference (P95)**
* Bootstrap-based confidence intervals

**Paper Reference:** Full experimental results and figures are included in the analysis document. 

---

## 2. Repository Structure

```
├── models/
│   └── Logistic_models/          # Trained LR models (various learning rates)
│
├── src/
│   ├── preprocess.py             # Dataset preprocessing utilities
│   ├── train_lr.py               # Logistic Regression training
│   ├── vader_eval.py             # VADER inference + metrics
│   ├── bert_jigsaw_eval.py       # BERT fine-tuning & bias testing
│   └── bias_metrics.py           # ASD, P95, paired-sentence evaluation
│
├── IMDB_Dataset.ipynb            # VADER evaluation notebook
├── Logistic_Regression_Analysis.ipynb
├── Jigsaw_Analysis.ipynb
├── Jigsaw_Toxicity_Dataset_Analysis.ipynb
├── .gitignore
└── requirements.txt
```

---

## 3. Datasets Used

| Dataset                            | Purpose                                    |
| ---------------------------------- | ------------------------------------------ |
| **IMDb Reviews**                   | Evaluate VADER accuracy & confusion matrix |
| **SST-2**                          | Train Logistic Regression / baseline BERT  |
| **Jigsaw Toxicity**                | Fine-tune BERT with biased data            |
| **Equity Evaluation Corpus (EEC)** | Standardized bias testing (gender, race)   |

---

## 4. Key Results (Summary)

### **4.1. VADER (IMDb)**

* Accuracy: **0.6973**
* F1 Score: **0.7381**
* Race bias ASD: **0.0229**
* Gender bias ASD: **0.0153**

Despite being rule-based, VADER still shows measurable polarity shifts.

---

### **4.2. Logistic Regression (SST-2)**

Best results at **learning rate 0.05**:

| Learning Rate | Test Accuracy | Test F1    |
| ------------- | ------------- | ---------- |
| **0.05**      | **0.7853**    | **0.7749** |

**Bias increases sharply with higher learning rates**
(P95 rises from ~0.04 → ~0.59 as LR increases).

---

### **4.3. BERT (Jigsaw Fine-tuned)**


* Shows **small**, **medium**, and **large** probability shifts depending on the slice.
* Example group differences:

  * Male = 0.5011 vs Female = 0.4914
  * Another slice: Male = 0.305 vs Female = 0.353
* ASD comparison plot:

  * Baseline BERT: **0.120**
  * Biased Jigsaw variant: **0.080**

---

## 5. Evaluation Metrics

All models use the EEC paired-sentence method:

* **ASD**: Average |score_A – score_B|
* **P95**: 95th percentile of |score_A – score_B|
* Bootstrap CIs where appropriate

This allows comparison across lexicon-based and contextual models.

---

## 6. Running the Code

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train Logistic Regression

```bash
python src/train_lr.py --lr 0.05
```

### Run VADER evaluation

```bash
python src/vader_eval.py
```

### Run BERT fine-tuning (Jigsaw)

```bash
python src/bert_jigsaw_eval.py
```

---

## 7. Reproducing Bias Evaluation

```bash
python src/bias_metrics.py --model lr --learning_rate 0.05
python src/bias_metrics.py --model vader
python src/bias_metrics.py --model bert_jigsaw
```

Outputs include:

* ASD values
* P95 values
* Distribution plots
* Per-group aggregated metrics

---

## 8. Paper Citation

If you use this work, cite:

```
Analyzing Inherent and Learned Bias in Sentiment Models:
A Comparative Study of Logistic Regression, VADER, and BERT.
```
