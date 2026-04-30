# 🛵 Food Delivery Time Predictor

> A Machine Learning web application that predicts food delivery time in minutes based on real-world delivery conditions — built with an **Optimized Random Forest Regressor** and deployed using **Streamlit**.

🔗 **Live Demo:** [food-delivery-time-prediction-rf.streamlit.app](https://food-delivery-time-prediction-rf.streamlit.app/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [EDA Highlights](#eda-highlights)
- [Models & Performance](#models--performance)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [App Features](#app-features)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [Requirements](#requirements)
- [Tech Stack](#tech-stack)

---

## 🧭 Overview

This project tackles a **regression problem** — predicting how long a food delivery will take (in minutes) based on multiple real-world factors including distance, weather, traffic level, time of day, vehicle type, preparation time, and courier experience.

The final model is an **Optimized Random Forest Regressor**, tuned using `RandomizedSearchCV`, achieving an **R² score of 0.79** on the test set.

---

## 📊 Dataset

| Property | Details |
|---|---|
| File | `Food_Delivery_Times.csv` |
| Rows | 1,000 |
| Total Columns | 9 |
| Target Variable | `Delivery_Time_min` |
| Dropped Column | `Order_ID` (not useful for prediction) |

### Features Used

| Feature | Type | Description |
|---|---|---|
| `Distance_km` | Numeric | Distance from restaurant to delivery point |
| `Weather` | Categorical | Clear, Rainy, Foggy, Snowy, Windy |
| `Traffic_Level` | Categorical | Low, Medium, High |
| `Time_of_Day` | Categorical | Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | Categorical | Bike, Scooter, Car |
| `Preparation_Time_min` | Numeric | Time taken by restaurant to prepare order |
| `Courier_Experience_yrs` | Numeric | Years of delivery experience |

---

## 🔄 Project Workflow

```
Raw CSV Data
    ↓
Data Cleaning & Missing Value Treatment
    ↓
Exploratory Data Analysis (EDA)
    ↓
Label Encoding (Categorical → Numeric)
    ↓
Train-Test Split
    ↓
Baseline Model (Linear Regression)
    ↓
Random Forest Regressor (Default)
    ↓
Hyperparameter Tuning (RandomizedSearchCV)
    ↓
Optimized Random Forest Model
    ↓
Model Saved as .pkl
    ↓
Streamlit App Deployment
```

---

## 🔍 EDA Highlights

- Handled missing values in `Weather`, `Traffic_Level`, `Time_of_Day` using **mode imputation**
- Handled missing values in `Courier_Experience_yrs` using **median imputation**
- Dropped `Order_ID` as it carries zero predictive information
- Generated **Correlation Matrix (Heatmap)** for numerical features
- Visualized distributions and relationships between features and target variable

---

## 📈 Models & Performance

### Baseline — Linear Regression

| Metric | Score |
|---|---|
| MAE | 10.56 |
| MSE | 226.82 |
| RMSE | 15.06 |
| **R²** | **0.49** |

### Default Random Forest Regressor

| Metric | Score |
|---|---|
| MAE | 7.06 |
| MSE | 100.12 |
| RMSE | 10.01 |
| **R²** | **0.78** |

### ✅ Optimized Random Forest Regressor *(Final Model)*

| Metric | Score |
|---|---|
| MAE | **6.88** |
| MSE | **92.53** |
| RMSE | **9.62** |
| **R²** | **0.79** |

> **Conclusion:** The Optimized Random Forest outperformed both the Linear Regression baseline and the Default RF model across all evaluation metrics.

---

## ⚙️ Hyperparameter Tuning

Used **`RandomizedSearchCV`** to search over the following parameter space:

| Parameter | Search Space |
|---|---|
| `n_estimators` | randint range |
| `max_features` | randint range |
| `max_depth` | randint range |
| `min_samples_split` | randint range |
| `min_samples_leaf` | randint range |

### 🏆 Best Parameters Found

```python
{
  'max_depth': 11,
  'max_features': 3,
  'min_samples_leaf': 3,
  'min_samples_split': 3,
  'n_estimators': 781
}
```

---

## ✨ App Features

- 🎛️ **Interactive sliders** for Distance and Preparation Time
- 🌦️ **Weather selector** — Clear, Rainy, Foggy, Snowy, Windy
- 🚦 **Traffic level** — Low, Medium, High
- 🕐 **Time of day** — Morning, Afternoon, Evening, Night
- 🛵 **Vehicle type** — Bike, Scooter, Car
- 👨‍💼 **Courier experience** slider (0–10 years)
- ⏱️ **Predicted delivery time** with ±5 min window
- 😄 **Contextual fun message** based on wait time
- 🎨 **Dark gradient UI** with purple theme

---

## 🗂️ Project Structure

```
├── main.py                    # Streamlit web app
├── Notebook.ipynb             # Full ML pipeline notebook (Google Colab)
├── optimized_rf_model.pkl     # Trained & tuned Random Forest model
├── label_encoders.pkl         # Saved LabelEncoders for categorical features
├── Food_Delivery_Times.csv    # Original dataset
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/food-delivery-predictor.git
cd food-delivery-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run main.py
```

App will open at `http://localhost:8501` in your browser.

---

## 📦 Requirements

```
streamlit
scikit-learn
numpy
pandas
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Seaborn & Matplotlib | EDA visualizations |
| Scikit-learn | ML models, encoding, tuning |
| Pickle | Model serialization |
| Streamlit | Web app deployment |
| Google Colab | Model training environment |

---

