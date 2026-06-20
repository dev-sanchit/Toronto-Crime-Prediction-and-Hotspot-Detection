

---

```markdown
# 📊 Toronto Crime Analytics AI: Predictive Policing System
> **An Integrated Computational Framework for Spatial Clustering and Additive Seasonal Forecasting**

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Facebook Prophet](https://img.shields.io/badge/Prophet-008080?style=for-the-badge&logo=meta&logoColor=white)](https://facebook.github.io/prophet/)

---

## 🌌 Project Space Overview
This project bridges the gap between raw data science prototyping and production-ready interactive deployments. It migrates complex mathematical modeling protocols from exploratory Jupyter Notebooks into a single-script, hyper-interactive frontend. 

By utilizing **Unsupervised Machine Learning** for regional hazard stratification and **Additive Regression Paradigms** for temporal forecasting, this system processes over 100,000 Major Crime Indicator (MCI) records in Toronto to deliver proactive, data-driven intelligence for public safety resource allocation.

---

## 🛠️ System Core Architecture


```

[ Raw Toronto MCI Dataset ]
│
▼
┌────────────────────┐
│ Cache Data Loader  │ ──► Drops Nulls & Standardizes Geometries
└────────────────────┘
│
┌──────┴──────────────────────┐
▼                             ▼
┌──────────────────┐         ┌────────────────────┐
│   K-Means Core   │         │ Prophet Forecasting│
├──────────────────┤         ├────────────────────┤
│ Scaled Vectors   │         │ Time-Series Signals│
│ PCA Component    │         │ Linear Growth      │
│ Reductions (2D)  │         │ Fourier Seasonality│
└──────────────────┘         └────────────────────┘
│                             │
└──────┬──────────────────────┘
▼
┌─────────────────────────────────────────────────┐
│ Hyper-Interactive Glassmorphic Streamlit UI     │
├─────────────────────────────────────────────────┤
│ 📂 Workspace Intro                              │
│ 📊 Multi-Dimensional Metric Dashboard            │
│ 🏘️ Unsupervised Spatial Segmentation Engineering │
│ 🔮 Chronological Geospatial Hotspot Engine     │
└─────────────────────────────────────────────────┘
```

---

## ✨ Advanced Interactive Features

### 1. Glassmorphic User Interface
* Custom CSS styles inject a dark cyber-aesthetic layout.
* Modern content containers smoothly shift position and trigger neon cyan boundary glows when hovered over.
* Re-engineered, smooth-scrolling fluid navigation interface linked dynamically to page routing.

### 2. Chronological Geospatial Hotspot Engine
* Interactive maps powered by WebGL render high-density vector coordinates seamlessly.
* Multi-dimensional data slicers let you segment crime data by **MCI Type, Calendar Year, Target Month,** and **Diurnal Hour Frames (0-23)** simultaneously.
* Precise hover-focused interaction highlights points instantly to reveal localized offense types and community data without cluttering the map.

### 3. Unsupervised Spatial Segmentation
* Uses `StandardScaler` normalization to prevent feature dominance across indicators.
* Performs real-time **Principal Component Analysis (PCA)** to project multi-dimensional matrices into clear 2D scatter plots.
* Exposes a hyperparameter slider to let users recalculate cluster allocations on the fly.

### 4. Algorithmic Forecasting Module
* Leverages **Facebook Prophet** to fit historical metrics against trend growth, yearly cycles, and daily variations.
* Computes and renders a continuous 12-month future prediction with a shaded 80% statistical confidence envelope.

---

## 🚀 Getting Started

### 📦 Installation
1. Clone the repository to your desktop environment:
   ```bash
   git clone [https://github.com/Ramlakahn/Crime-prediction-and-hotspot-detection.git](https://github.com/Ramlakahn/Crime-prediction-and-hotspot-detection.git)
   cd Crime-prediction-and-hotspot-detection

```

2. Install the necessary mathematical and visualization libraries:
```bash
pip install streamlit pandas numpy matplotlib seaborn prophet scikit-learn plotly

```



### 🗃️ Data Placement

Ensure that your file repository structure matches the following setup precisely, keeping the dataset file in your working folder:

```text
/Crime-prediction-and-hotspot-detection
├── app.py
├── MCI_2014_to_2019.csv
├── Crime Prediction_clustering.ipynb
└── Crime_Pred_analysis.ipynb

```

### ⚡ Run the Application

Start the Streamlit runtime instance using your system terminal:

```bash
streamlit run app.py

```

---

## 📊 Individual Evaluation Deliverables

### Results Summary

* **Spatial Heterogeneity:** K-Means modeling paired with PCA effectively segments neighborhoods into distinct risk categories, proving that urban crime concentrates heavily in specific hubs rather than spreading uniformly.
* **Temporal Cyclicality:** Time-series decomposition reveals strong seasonal fluctuations, capturing reproducible surge intervals during summer transitions and specific times of day.
* **Predictive Accuracy:** The additive regression model aligns tightly with historical verification sets, generating actionable 12-month outlook boundaries with statistical confidence metrics.

### Conclusion

This project demonstrates the value of migrating standalone machine learning prototypes into functional data applications. By combining unsupervised spatial grouping with predictive seasonal modeling, this platform builds a reliable computational roadmap for predictive policing. This structure provides a scalable framework to help administrative agencies transition from reactive scheduling to data-backed, proactive public safety management.

---





