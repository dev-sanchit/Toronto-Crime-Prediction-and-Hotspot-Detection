# 🚔 Toronto Crime Analytics AI

<p align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Prophet](https://img.shields.io/badge/Prophet-Time_Series-green)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)

</p>

> **An interactive crime analytics platform that combines unsupervised learning and time-series forecasting to identify crime hotspots and predict future crime trends across Toronto.**

This project analyzes Toronto Major Crime Indicator (MCI) records and transforms raw crime data into actionable insights through clustering, forecasting, and geospatial visualization.

---

## 🎯 Project Overview

Urban crime patterns are influenced by both **location** and **time**.

This project explores these patterns using:

* **K-Means Clustering** to identify crime-prone regions
* **Principal Component Analysis (PCA)** for visualization
* **Facebook Prophet** for forecasting future crime activity
* **Interactive Maps** for hotspot exploration
* **Streamlit** for deployment as a web application

The goal is to demonstrate how machine learning and data analytics can be used to support data-driven public safety planning.

---

## ✨ Features

### 📍 Crime Hotspot Detection

* Interactive crime maps
* Filtering by crime category
* Filtering by year and month
* Hour-wise crime analysis
* Community-level hotspot visualization

### 🤖 Unsupervised Learning

* Feature scaling using StandardScaler
* K-Means clustering
* Dynamic cluster selection
* PCA dimensionality reduction
* Visual cluster exploration

### 📈 Crime Forecasting

* Time-series forecasting with Prophet
* Seasonal trend analysis
* Future crime projections
* Confidence interval visualization
* Historical vs predicted comparisons

### 🎨 Interactive Dashboard

* Built using Streamlit
* Interactive charts and maps
* Real-time filtering
* User-friendly exploration interface

---

## 🏗️ System Architecture

```text
Toronto Crime Dataset
          │
          ▼
Data Cleaning & Preprocessing
          │
          ▼
 ┌──────────────────────┐
 │ Feature Engineering  │
 └──────────────────────┘
          │
          ▼
 ┌──────────────────────┐
 │ K-Means Clustering   │
 └──────────────────────┘
          │
          ▼
 ┌──────────────────────┐
 │ PCA Visualization    │
 └──────────────────────┘
          │
          ▼
 ┌──────────────────────┐
 │ Prophet Forecasting  │
 └──────────────────────┘
          │
          ▼
 Interactive Streamlit Dashboard
```

---

## 📂 Project Structure

```text
Crime-prediction-and-hotspot-detection
│
├── app.py
├── MCI_2014_to_2019.csv
├── Crime Prediction_clustering.ipynb
├── Crime_Pred_analysis.ipynb
└── README.md
```

---

## 🧠 Machine Learning Components

### K-Means Clustering

Used to group regions with similar crime characteristics.

**Input Features**

* Geographic indicators
* Crime frequencies
* Community statistics

**Purpose**

* Identify high-risk zones
* Discover hidden spatial patterns
* Support hotspot detection

---

### Principal Component Analysis (PCA)

Used to reduce dimensionality and visualize clustering results.

**Benefits**

* Simplifies high-dimensional data
* Enables cluster visualization
* Improves interpretability

---

### Prophet Forecasting

Used for crime trend prediction.

**Forecast Components**

* Long-term trends
* Seasonal effects
* Historical crime patterns

**Output**

* 12-month future forecast
* Confidence intervals
* Trend visualization

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/Ramlakahn/Crime-prediction-and-hotspot-detection.git

cd Crime-prediction-and-hotspot-detection
```

### Install Dependencies

```bash
pip install streamlit pandas numpy matplotlib seaborn prophet scikit-learn plotly
```

### Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📊 Dataset

The project uses Toronto's **Major Crime Indicators (MCI)** dataset containing:

* Assault
* Auto Theft
* Break & Enter
* Robbery
* Theft Over

The dataset includes over 100,000 crime records with spatial and temporal attributes.

---

## 📈 Key Insights

* Crime incidents are spatially concentrated rather than uniformly distributed.
* Distinct crime clusters emerge across different communities.
* Seasonal crime patterns can be observed over time.
* Forecasting models capture recurring trends and provide future estimates of crime activity.

---

## 🛠️ Tech Stack

| Category         | Tools                       |
| ---------------- | --------------------------- |
| Programming      | Python                      |
| Dashboard        | Streamlit                   |
| Machine Learning | Scikit-Learn                |
| Forecasting      | Prophet                     |
| Data Processing  | Pandas, NumPy               |
| Visualization    | Plotly, Matplotlib, Seaborn |

---

## ⚠️ Limitations

* Forecast quality depends on historical data quality.
* K-Means clustering is sensitive to the selected number of clusters.
* Forecasts should be interpreted as analytical estimates rather than operational predictions.
* The system is intended for educational and analytical purposes.

---

## 👨‍💻 Author

Sanchit Srivastava

B.Tech Computer Science & Engineering
Bennett University

Interests: Machine Learning, Data Science, AI Engineering, Predictive Analytics

---

### ⭐ If you found this project interesting, consider starring the repository.
