# 📊 Thai Stock Halal Classifier – CRISP-DM Based README

This project classifies Thai stocks as **Halal** or **Haram** based on Islamic financial principles. The classification is powered by vocabulary screening, data mining, and a machine learning model trained on Thai stock metadata.

---

## 📘 Project Summary

* **Goal**: Automatically identify whether a Thai stock is Halal or Haram based on its description, industry, and sector.
* **Approach**: Use CRISP-DM methodology to build a machine learning pipeline for classification.

---

## 🚀 CRISP-DM Process Overview

### 1. Business Understanding

> Objective: Help Muslim investors identify stocks that comply with Islamic finance principles (Halal) and avoid Haram industries.

Questions to answer:

* Is this stock compliant with Islamic principles?
* Can a machine predict Halal/Haram based on metadata?

---

### 2. Data Understanding

* Data Source: **Apache Cassandra** database (`thai_stocks` table)
* Supplemented by: `yfinance` API (optionally)
* Key Fields: `ticker`, `name`, `sector`, `industry`, `description`

Example Entry:

| ticker | name               | sector   | industry  | description                    |
| ------ | ------------------ | -------- | --------- | ------------------------------ |
| PTT.BK | PTT Public Company | Energy   | Oil & Gas | National oil and gas company   |
| SABINA | Sabina Co., Ltd.   | Consumer | Apparel   | Manufactures women’s underwear |

---

### 3. Data Preparation

* Combine all descriptive fields into one (`combined_text`)
* Clean text (lowercase, strip nulls)
* Apply **Halal/Haram rules** using a predefined `vocab` dictionary:

```python
vocab = {
    "Interest": {
        "Finance": ["interest rate", "loan", "mortgage", "investment"]
    },
    "Gambling": ["casino", "bet", "poker"],
    "Pigs": ["pork", "swine", "pig"],
    "Music": ["music", "lyrics", "band"],
    "Prostitution": ["sex work", "brothel", "escort"]
}
```

If any forbidden words appear, the stock is labeled as `"Haram"`, otherwise `"Halal"`.

---

### 4. Modeling

* Technique: **Random Forest Classifier**
* Pipeline: `CountVectorizer` → `RandomForestClassifier`
* Train/Test Split: 80/20
* Input: `combined_text` (from metadata)
* Output: `halal_status` (Halal / Haram)

---

### 5. Evaluation

* Basic metric: **Accuracy** on test set
* Additional checks: Manual keyword-based evaluation

Example Result:

```python
Accuracy on test set: 94.5%
```

---

### 6. Deployment

* Created function:

```python
def predict_halal_status(stock_name: str) -> str
```

* Accepts input like company name or description
* Returns prediction: **"Halal"** or **"Haram"**

Example:

```python
predict_halal_status("beer and wine production")  → "Haram"
predict_halal_status("SABINA")                    → "Halal"
```

---

## 📦 Project Structure

```bash
.
├── model.ipynb             # Main Jupyter Notebook
├── README.md               # This document
└── requirements.txt        # Python dependencies (optional)
```

---

## 🔮 Future Improvements

* Improve classifier with deep learning models (BERT, LSTM)
* Add web-based user interface with Gradio or Streamlit
* Connect live `yfinance` updates for real-time predictions

