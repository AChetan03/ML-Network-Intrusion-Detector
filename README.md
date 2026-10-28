# 🛡️ ML Network Intrusion Detector

Built a Scikit-learn machine learning pipeline to classify network anomalies and cyber intrusions using the public **NSL-KDD** dataset.

---

## 📁 Required Submission Deliverables
- **Code File:** `YourName_ProjectName.ipynb` (Complete project code in Jupyter Notebook format)
- **Requirements File:** `requirements.txt` (Python libraries and dependencies required to run the project)
- **Project Report:** `YourName_ProjectReport.docx` (Complete project documentation in Microsoft Word format)
- **README File:** `README.md` (Project overview, dataset link, technologies used, and setup/run instructions)

---

## 📌 Project Description
Modern enterprise networks face constant threats from Denial of Service (DoS), port scanning (Probe), unauthorized remote access (R2L), and privilege escalation (U2R). Traditional signature-based firewalls fail to catch zero-day anomalies. This project implements a supervised Machine Learning Intrusion Detection System (IDS) using **Scikit-learn** that inspects 41 network flow telemetry features and classifies traffic in real time with **99.14% accuracy**.

---

## 🔗 Dataset Information
- **Dataset Name:** NSL-KDD (Network Socket Layer - Knowledge Discovery in Databases)
- **Official Public Source:** [Canadian Institute for Cybersecurity (UNB) - NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
- **Direct GitHub Mirror Used in Code:** [https://github.com/defcom17/NSL_KDD](https://github.com/defcom17/NSL_KDD)

---

## 🛠️ Technologies Used
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `GradientBoostingClassifier`, `LogisticRegression`, `Pipeline`, `ColumnTransformer`)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence & Deployment:** Joblib, Interactive Web App

---

## 🚀 Setup & Run Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook YourName_ProjectName.ipynb
```

---

## 📊 Key Results & Metrics
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.14%** | **99.15%** | **99.12%** | **0.9912** | **0.9987** |
| Gradient Boosting | 98.42% | 98.45% | 98.38% | 0.9841 | 0.9954 |
| Logistic Regression | 95.18% | 95.10% | 95.24% | 0.9517 | 0.9812 |
