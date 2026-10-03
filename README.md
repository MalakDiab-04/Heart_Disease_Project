# ❤️ Heart Disease Risk Prediction & Analysis

A complete end-to-end Machine Learning pipeline and interactive Streamlit web application for analyzing, predicting, and clustering heart disease risks using the **Heart Disease UCI Dataset**.

---

## 📌 Project Overview

Heart disease remains one of the leading global causes of death. This project applies end-to-end data science methodology to explore risk factors, select optimal features, apply dimensionality reduction (PCA), train supervised classification models, perform unsupervised clustering, optimize model hyperparameters, export a reproducible ML pipeline, and deploy an interactive web interface.

### Key Objectives
* **Data Preprocessing & Cleaning**: Imputation, categorical encoding, scaling, and Exploratory Data Analysis (EDA).
* **Dimensionality Reduction**: Principal Component Analysis (PCA) to retain maximum variance with fewer dimensions.
* **Feature Selection**: Evaluate feature importance using Random Forest, XGBoost, Recursive Feature Elimination (RFE), and Chi-Square tests.
* **Supervised Learning**: Train & compare Logistic Regression, Decision Trees, Random Forest, and Support Vector Machines (SVM).
* **Unsupervised Learning**: K-Means clustering (elbow method) and Hierarchical Clustering (dendrogram analysis).
* **Hyperparameter Tuning**: Optimize classifiers using `GridSearchCV` and `RandomizedSearchCV`.
* **Deployment & UI**: Interactive Streamlit application (`ui/app.py`) deployed locally and exposed via Ngrok.

---

## 📁 Repository Structure

```text
Heart_Disease_Project/
│── data/
│   ├── heart_disease.csv          # Raw UCI dataset
│   └── processed/                 # Train/test split artifacts
│── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│── models/
│   ├── final_model.pkl            # Final production pipeline model
│   └── extra_models/              # Baseline and intermediate checkpoints
│── ui/
│   ├── app.py                     # Streamlit web application
│── deployment/
│   ├── ngrok_setup.txt            # Deployment instructions & live link
│── results/
│   ├── evaluation_metrics.txt     # Complete evaluation summary
│   ├── figures/                   # Generated charts & plots
│   └── tables/                    # CSV performance metrics
│── README.md
│── requirements.txt
└── .gitignore
```

---

## 📊 Model Evaluation & Results

### Supervised Classification Baseline vs. Tuned Models

| Model | Baseline Accuracy | Tuned Accuracy | Tuned Precision | Tuned Recall | Tuned F1-Score | Tuned ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **83.61%** | **86.89%** | **85.71%** | **85.71%** | **85.71%** | **0.9297** |
| **Decision Tree** | 86.89% | 88.52% | 88.89% | 85.71% | 87.27% | 0.8452 |
| **Random Forest** | 86.89% | 85.25% | 80.65% | 89.29% | 84.75% | 0.9491 |
| **SVM** | 86.89% | 86.89% | 85.71% | 85.71% | 85.71% | 0.9297 |

> **Selected Final Model**: Tuned **Logistic Regression** pipeline exported to `models/final_model.pkl` due to robust cross-validation AUC (0.908) and high overall stability across metrics.

### Unsupervised Clustering Summary
* **K-Means Clustering**: Optimal $K=4$ clusters determined via the Elbow Method (Silhouette Score: `0.1460`).
* **Hierarchical Clustering**: Dendrogram distance threshold identified 2 main clusters (Silhouette Score: `0.1649`).

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and install required dependencies:

```bash
git clone https://github.com/MalakDiab-04/Heart_Disease_Project.git
cd Heart_Disease_Project
pip install -r requirements.txt
```

### 2. Running Jupyter Notebooks

Execute notebooks sequentially to reproduce data cleaning, feature engineering, modeling, and evaluation:

```bash
jupyter lab
# Open notebooks in order: 01 -> 02 -> 03 -> 04 -> 05 -> 06
```

### 3. Launching the Streamlit UI

To run the web app locally:

```bash
python -m streamlit run ui/app.py --server.port 8501
```

Access the dashboard in your browser at `http://localhost:8501`.

---

## 🌐 Public Deployment (Ngrok)

To share the application over a public Ngrok tunnel:

1. **Start Streamlit App**:
   ```bash
   python -m streamlit run ui/app.py --server.port 8501
   ```
2. **Start Ngrok Tunnel** (in a separate terminal):
   ```bash
   ngrok http 8501
   ```
3. **Access Public Link**: Copy the generated `https://<subdomain>.ngrok-free.dev` URL provided in the terminal.

> Details and live link guidelines are available in [`deployment/ngrok_setup.txt`](deployment/ngrok_setup.txt).

---

## 🛠️ Tech Stack & Libraries

* **Language**: Python 3.10+
* **Data & Math**: Pandas, NumPy, SciPy
* **Machine Learning**: Scikit-Learn, XGBoost, Joblib
* **Visualization**: Matplotlib, Seaborn
* **Web UI & Deployment**: Streamlit, Pyngrok

---

## 📄 License & Acknowledgments

* **Dataset Source**: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) (`data/heart_disease.csv`).
* Educational and research project built for machine learning experimentation.