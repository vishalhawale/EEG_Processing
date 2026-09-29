# CHB-MIT EEG Seizure Detection & Localization Pipeline

An end-to-end Machine Learning and Deep Learning pipeline designed to detect epilepsy seizure onset and localize focal origins across multi-channel EEG signals from the benchmark **CHB-MIT Scalp EEG Database**.

## Project Overview

This repository provides a modular, cross-subject benchmark evaluation comparing **Linear Models**, **Tree-Based Ensembles**, and **1D Convolutional Neural Networks (PyTorch)**.

### Key Technical Features

* **Standardized 18-Channel Bipolar Montage:** Standardized channel mapping adhering to the International 10–20 System.

* **Signal Preprocessing:** Bandpass filtering ($0.5-40\text{ Hz}$) and Notch filtering ($50\text{ Hz}$) using `MNE-Python`.

* **Feature Extraction:** 162 total features extracted across 18 channels combining time-domain metrics (Line Length, Mean, Variance) and spectral power bands ($\delta, \theta, \alpha, \beta, \gamma$).

* **Data Imbalance Handling:** Oversampling via **SMOTE** on training signals to mitigate interictal-ictal skewness.

* **Cross-Subject Generalization:** Models are trained on Subject `chb01` and evaluated on unseen Subject `chb02`.

* **Seizure Localization:** Channel importance ranking to identify localized epileptogenic zones.

## Cross-Subject Performance Benchmark (Test Subject: `chb02`)

Evaluating models across subjects highlights the real-world performance trade-offs in clinical alarm systems:

| Model Architecture | Accuracy (%) | ROC-AUC | Recall (Sensitivity) | Precision | F1-Score | 
 | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Random Forest** | **99.41%** | **0.9794** | 26.32% | **0.7500** | 0.3896 | 
| **1D-CNN (PyTorch)** | 94.01% | 0.9507 | **85.96%** | 0.1175 | **0.2068** | 
| **Gradient Boosting** | 94.81% | 0.9047 | 29.82% | 0.0400 | 0.0705 | 
| **Logistic Regression** | 42.92% | 0.7898 | **92.98%** | 0.0218 | 0.0426 | 

### Clinical Insights

* **The Accuracy Paradox:** While **Random Forest** achieves high accuracy (99.41%), it misses **73.78%** of actual seizure windows due to subtle amplitude changes between subjects.

* **Clinical Screening Preference:** The **1D-CNN** provides the best operational balance, successfully detecting **85.96% of all seizure windows** while maintaining high ROC-AUC ($0.9507$).

## 📁 Repository Structure

```
EEG_Processing/
├── data/                           # CHB-MIT raw EDF files & summary text
├── artifacts/                      # Features (.npy) & benchmark figures
├── src/                            # Modular package source code
│   ├── data_loader.py              # EDF parsing & channel mapping
│   ├── feature_extraction.py       # Sliding window feature computation
│   ├── machine_learning_models.py  # PyTorch CNN & classical classifiers
│   └── evaluation_and_visualise.py # Evaluation metrics & visualization
├── run_pipeline.py                 # Pipeline execution entry point
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation

```

## Getting Started

### 1. Clone & Set Up Environment

```
git clone https://github.com/your-username/EEG_Processing.git
cd EEG_Processing

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

### 2. Download Data

Place the CHB-MIT EDF files for `chb01` and `chb02` along with their summary files into the `data/` directory.

### 3. Run Pipeline

```
python run_pipeline.py

```