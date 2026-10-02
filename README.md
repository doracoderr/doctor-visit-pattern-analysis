# Doctor Visit Pattern Analysis

AICTE TIRTC DIY Project

**Student:** Abhishek Prasad
**College:** Dronacharya Government College
**AICTE STU ID:** STU6a31fc7ff2ba91781660799

## About
A data analytics study on 5,190 patient records linking personal details, health status, income and healthcare coverage with how often people visit a doctor.

## What is done
- Data cleaning and encoding
- Correlation heatmap and group comparisons (age, gender, chronic condition, coverage)
- Logistic Regression (odds ratios) and Gradient Boosting to predict whether a person visits a doctor

## Key results
- 79.8% of people had zero visits
- Reduced-activity days and illness count raise visit odds the most
- Share of people with 1+ visit rises from 14.7% (age 25 or younger) to 30.2% (65+)
- Logistic Regression: 73.3% accuracy, AUC 0.763, recall 67%
- Gradient Boosting: 82.4% accuracy, AUC 0.770, recall 30%

## Files
- `doctor_visits_analysis.ipynb`: full notebook with outputs
- `doctor_visits_analysis.py`: same code as a script
- `1776250375-P2-Healthcare_Analytics_for_Doctor_Visits__1_.csv`: dataset
- `requirements.txt`: Python libraries

## How to run
```
pip install -r requirements.txt
jupyter notebook doctor_visits_analysis.ipynb
```
Or open the notebook in Google Colab and upload the CSV to the same folder.
