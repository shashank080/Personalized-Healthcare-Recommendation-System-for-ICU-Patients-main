# Personalized-Healthcare-Recommendation-System-for-ICU-Patients

A machine learning-based clinical decision support tool for predicting ICU admission, estimating patient length of stay, and classifying patient risk levels based on MIMIC-III data.

---

## Project Description

This project aims to support ICU triage and planning by using both structured and unstructured data to:

- Predict whether a patient will be admitted (classification)
- Estimate length of ICU stay (regression)
- Classify admitted patients into Low/Medium/High risk categories
- Perform clustering on clinical notes to extract thematic groupings

Built using real-world ICU data from the **MIMIC-III 10k** dataset and deployed via a **Streamlit web app**.

---

## Dataset

- **Name**: [MIMIC-III 10k Subset](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k)
- **Source**: MIT Lab for Computational Physiology (via PhysioNet / Kaggle)
- **Records**: 10,000 ICU patients
- **Features**: Vitals, demographics, diagnoses, clinical notes

---

## ML Models Used

| Model              | Use Case                  | Type           |
|-------------------|---------------------------|----------------|
| Logistic Regression | Admission (baseline)      | Classification |
| Random Forest      | Admission, LOS             | Classification & Regression |
| XGBoost            | Admission (comparative)    | Classification |
| KMeans + TF-IDF    | NLP Clustering (Notes)     | Unsupervised   |

---

## Evaluation Summary

- **Best Model**: Random Forest  
- **Accuracy**: 81%  
- **F1-Score**: 0.82  
- **ROC AUC**: 0.863  
- **Validation**: 10-Fold Cross-Validation + 20% Hold-Out Test

---

## Risk Stratification Logic

- Low Risk: Probability < 0.4  
- Medium Risk: 0.4 ≤ p < 0.7  
- High Risk: p ≥ 0.7

---

## NLP Pipeline

- **Text Source**: NOTEEVENTS.csv  
- **Pipeline**: Text cleaning → TF-IDF → Truncated SVD → KMeans (k=5)  
- **Goal**: Cluster clinical notes to uncover thematic patterns among ICU patients

---

## Web App

A Streamlit-based interface for real-time predictions:

- Enter vitals, demographics, and diagnoses
- Get predictions for:
  - Admission status
  - Estimated ICU stay duration
  - Risk level classification
---

## References

1. Johnson, A.E.W. et al. (2016). MIMIC-III Database. Scientific Data. [Link](https://physionet.org/content/mimiciii/1.4/)
2. Kaggle Dataset: [MIMIC-III 10k Subset](https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k)
3. Rajkomar, A. et al. (2019). Machine Learning in Medicine. NEJM.
4. Frontiers in Public Health (2022). ICU Risk Prediction Using ML & EHR

---

## How to Run the App

```bash
pip install -r requirements.txt
streamlit run app/app.py





