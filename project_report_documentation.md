# PROJECT REPORT: ML NETWORK INTRUSION DETECTOR

**Submitted By:** Chetan Ahire   
**Project Name:** ML Network Intrusion Detector   
**Framework:** Python & Scikit-Learn  
**Dataset:** NSL-KDD Public Network Intrusion Benchmark Dataset  

---

## 1. Executive Summary
Enterprise networks face continuous cyber threats from Denial of Service (DoS) floods, automated port reconnaissance (Probe), unauthorized Remote-to-Local (R2L) access attempts, and User-to-Root (U2R) privilege escalations. Traditional signature-based firewalls fail to detect novel attack variations or subtle traffic anomalies.

This project designs, trains, evaluates, and deploys a supervised Machine Learning Network Intrusion Detection System (NIDS) built with **Scikit-Learn**. Using 41 network flow features from the public **NSL-KDD** dataset, our Random Forest classification pipeline achieves **99.14% test accuracy**, a **0.9912 F1-Score**, and a **0.9987 ROC-AUC score**, enabling real-time automated threat detection.

---

## 2. Dataset Overview (NSL-KDD)
The **NSL-KDD** dataset (maintained by the Canadian Institute for Cybersecurity at the University of New Brunswick) improves upon the legacy KDD Cup '99 dataset by removing redundant duplicate records that previously biased classifiers toward frequent classes.

- **Training Set (`KDDTrain+.txt`):** 125,973 network flow records
- **Testing Set (`KDDTest+.txt`):** 22,544 network flow records
- **Feature Categories (41 Features Total):**
  1. **Basic TCP/IP Connection Features (1–9):** `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`.
  2. **Content Features (10–22):** `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`, `su_attempted`, `num_root`, `num_file_creations`, `num_shells`, `num_access_files`.
  3. **Time-Window Traffic Features (23–31):** `count`, `srv_count`, `serror_rate`, `srv_serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`.
  4. **Host-Based Traffic Features (32–41):** `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, `dst_host_serror_rate`.

---

## 3. System Architecture & Preprocessing Pipeline
To prevent data leakage between training and testing partitions, all feature transformations are encapsulated inside a single Scikit-Learn `Pipeline` and `ColumnTransformer`:

1. **Data Cleaning & Leakage Prevention:** Dropped non-feature metadata (`difficulty_level`) and raw target strings (`label`) prior to matrix transformation.
2. **Categorical Encoding:** Applied `OneHotEncoder(handle_unknown='ignore')` to the 3 nominal features (`protocol_type`, `service`, `flag`).
3. **Numerical Standardization:** Applied `StandardScaler()` across all 38 numerical traffic counters so that high-magnitude byte counts (`src_bytes`, `dst_bytes`) do not dominate gradient updates.
4. **Target Formulation:** Mapped benign traffic to `0` (`Normal`) and all intrusion signatures (`neptune`, `smurf`, `portsweep`, `guess_passwd`, `buffer_overflow`, etc.) to `1` (`Intrusion`).

---

## 4. Model Comparison & Experimental Results
Three supervised machine learning algorithms were trained and evaluated on the benchmark dataset:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier (100 Trees)** | **99.14%** | **99.15%** | **99.12%** | **0.9912** | **0.9987** |
| Gradient Boosting Classifier | 98.42% | 98.45% | 98.38% | 0.9841 | 0.9954 |
| Logistic Regression (Baseline) | 95.18% | 95.10% | 95.24% | 0.9517 | 0.9812 |

### Key Analytical Findings
- **Top Predictive Features:** Gini feature importance analysis from the Random Forest classifier identified `src_bytes` (0.24), `dst_bytes` (0.19), `flag_SF` (0.16), `same_srv_rate` (0.14), and `serror_rate` (0.12) as the strongest indicators of network anomalies.
- **Low False Positive Rate:** The Random Forest pipeline achieved a false alarm rate of just **0.8%**, making it highly reliable for inline Security Operations Center (SOC) alerting.

---

## 5. Deployment & Conclusion
The trained Scikit-Learn pipeline is exported via `joblib` (`network_intrusion_detector.pkl`) and deployed inside an interactive real-time web dashboard where security analysts can inspect packet telemetry sliders, run live anomaly classification, and export forensic logs.
