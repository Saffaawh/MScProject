# MRI Appointment Scheduling System - MSc Data Science Project

This repository contains the source code for my MSc Data Science project. The system uses Machine Learning to optimize MRI appointment scheduling by predicting patient complexity and scan durations.

The codebase is structured around 4 main objectives:

## 📂 Objective 1: Data Creation & Preparation
This module handles the simulation and processing of medical appointment data.
* **Dataset Generation:** Code to simulate two primary synthetic datasets:
    1.  **Historical Appointment Dataset:** Ground truth data for model training.
    2.  **MRI Request Dataset:** A pool of new appointment requests.
* **Sampling:** Includes a stratified sample of 500 requests used for testing the scheduler.
* **Analysis:** Contains feature engineering scripts and Exploratory Data Analysis (EDA) to break down dataset distributions.

## 📂 Objective 2: Classification (Risk Assessment)
This module compares multiple algorithms to classify patient risk.
* **Models Implemented:** * **Decision Tree Classifier (DTC):** Baseline model for interpretability.
    * **Random Forest Classifier (RFC):** Advanced ensemble model for higher accuracy.
* **Analysis:** Includes comparative analysis, confusion matrices, and feature importance evaluation to select the best-performing model for the scheduler.

## 📂 Objective 3: Regression (Time Prediction)
This module focuses on the **Regression Model**.
* Predicts the exact duration (or overrun minutes) for each appointment.
* Outputs evaluation metrics comparing predicted times against actual historical times.

## 📂 Objective 4: Scheduling System
The final implementation of the scheduling engine.
* **Baseline Scheduler:** Uses standard heuristics (averages) to book appointments.
* **ML Scheduler:** Uses the AI models from Objectives 2 & 3 to dynamically allocate slots.
* Includes visualization tools to compare the efficiency of both schedules.