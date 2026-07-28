# MRI Appointment Scheduling System

> **MSc Data Science & AI Final Project**  
> An end-to-end Machine Learning and dynamic scheduling pipeline designed to optimize MRI appointment booking, predict scan durations, and reduce schedule overruns.

---

## 📌 Executive Summary

Diagnostic imaging bottlenecks frequently result in delayed patient care and underutilized scanner capacity. This project implements a data-driven approach to MRI slot allocation. By combining classification models (to assess patient risk/complexity) with regression models (to predict precise scan durations), the system dynamically schedules requests to minimize delay and optimize throughput.

---

## 🛠 Tech Stack & Architecture

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Decision Tree Classifier, Random Forest Classifier, Regression Models)
* **Data Processing & Analysis:** Pandas, NumPy
* **Visualization & Interactive UI:** HTML5, Matplotlib

---

## 📂 Project Structure & Core Modules

### 📂 Objective 1: Data Creation & Preparation
Handles the simulation, pipeline ingestion, and processing of clinical appointment data.
* **Dataset Generation:** Simulates two synthetic datasets:
  * *Historical Appointment Dataset:* Ground truth data for model training and calibration.
  * *MRI Request Dataset:* Incoming pool of unscheduled appointment requests.
* **Sampling:** Implements stratified sampling (500 requests) for scheduler validation.
* **Exploratory Data Analysis (EDA):** Scripts for feature engineering and dataset distribution analysis.

### 📂 Objective 2: Classification (Risk Assessment)
Evaluates multiple machine learning algorithms to classify patient risk and appointment complexity.
* **Models Evaluated:** Decision Tree Classifier (DTC) baseline vs. Random Forest Classifier (RFC) ensemble.
* **Evaluation:** Feature importance metrics, confusion matrices, and hyperparameter tuning to handle class imbalance.

### 📂 Objective 3: Regression (Scan Time Prediction)
Focuses on predicting precise appointment durations and potential overruns.
* Predicts expected scan duration per patient based on historical operational metrics.
* Evaluates error metrics (MAE, RMSE) comparing ML predictions against standard static averages.

### 📂 Objective 4: Dynamic Scheduling Engine & UI
The core scheduling optimization engine featuring interactive visual dashboards.
* **Baseline Scheduler:** Standard heuristic booking using fixed average time blocks.
* **ML Dynamic Scheduler:** Integrates Objective 2 & 3 outputs to dynamically allocate scanner slot blocks.
* **Visual Scheduler Dashboards:** Interactive HTML/UI visualization tools to compare baseline vs. ML-optimized schedule efficiency and scanner occupancy.

---

## 🚀 Key Results & Insights

* **Overrun Reduction:** The ML-driven scheduler significantly reduced schedule drift compared to standard static time blocks.
* **Operational Impact:** Demonstrates how predictive model metrics translate into actionable capacity management for clinical delivery teams.
