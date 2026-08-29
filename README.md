# NetShield

### AI-Powered Network Intrusion Detection System

NetShield is a Machine Learning-based **Network Intrusion Detection System (NIDS)** that detects and classifies malicious network traffic using a two-tier **XGBoost** architecture.

## System Architecture

```text
                    Network Traffic
                           │
                           ▼
                ┌─────────────────────┐
                │ Data Preprocessing  │
                │ & Feature Selection │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │      Tier 1      │
                 │ Binary Classifier│
                 │     XGBoost      │
                 └─────────┬────────┘
                           │
                    ┌──────┴──────┐
                    │             │
                  Normal        Attack
                    │             │
                    ▼             ▼
              Normal Traffic ┌───────────────────┐
                             │      Tier 2       │
                             │ Multiclass        │
                             │ Classifier        │
                             │ XGBoost            │
                             └─────────┬─────────┘
                                       │
                                       ▼
                              Attack Classification
```

### Tier 1 — Binary Classification

Determines whether network traffic is **Normal** or an **Attack**.

**Performance**

* Accuracy: **99.25%**
* ROC-AUC: **0.9992**

### Tier 2 — Multiclass Classification

Classifies detected attacks into **9 attack categories**, including Exploits, Fuzzers, DoS, Reconnaissance, and others.

## Dataset & Processing

NetShield uses the **UNSW-NB15** dataset.

The preprocessing pipeline includes:

* Duplicate removal
* Missing-value handling
* Data type optimization
* Feature selection
* Train/test preparation

**Key results:**

* **480,633 duplicate rows removed**
* **51.2% reduction in RAM usage**
* IP and Port features excluded to improve generalization

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn · Streamlit · Parquet**

## Project Structure

```text
NetShield/
├── app/
├── models/
├── src/
├── requirements.txt
├── run_phase1.py
├── run_phase2.py
├── run_phase3.py
├── run_phase4.py
├── run_phase5.py
├── run_phase6.py
└── run_phase7.py
```

## Run the Application

```bash
pip install -r requirements.txt
streamlit run app/app.py
```



## Future Improvements

* Real-time network traffic detection
* SHAP-based model explainability
* FastAPI integration
* Docker containerization
* Cloud deployment

---

**NetShield — Detect. Classify. Protect.**
