# [Mold breakout prediction based on computer vision and machine learning]

## Overview
This project seeks to predict sticking breakouts using machine learning models. We employ the AdaBoost and Logistic Regression models to analyze and identify feature vectors extracted from thermographic images of abnormal sticking regions. The primary focus is to accurately distinguish abnormal temperature patterns, significantly reducing false positives while ensuring no sticking breakouts are missed. Consequently, this approach provides valuable insights for anomaly detection and prediction in the continuous casting process.

Link to this paper: https://link.springer.com/article/10.1007/s42243-024-01198-2

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)

## Installation
### Prerequisites
Before running the project, make sure you have the following dependencies installed:

- Python: version 3.6 or higher.
- Required Python package including numpy, pandas, matplotlib, sklearn etc,.


## Usage
### Running the Model
To run the model, follow these steps:

1. Download all files from this repository and ensure they are placed within the same root directory.

2. Open a terminal, navigate to the directory containing these files, and execute the appropriate script based on the model you want to test:

    To test the Adaboost model, run:
   python Feature_vectors_classification_by_Adaboost.py

    To test the Logisitic regression model, run:
   python Feature_vectors_classification_by_Logistic_Regression.py

4. Once the script executes, it will initiate a loop that iterates over various feature combinations with a fixed dimension. You can modify the number of features and their corresponding combinations in the script to meet your specific requirements. Ultimately, CSV files will be generated, containing the optimal feature combinations for the selected model.

5. Note: We apologize for including some code written in Chinese; we are working to provide a fully English version in future updates.

## Project Structure
The project is organized as follows:

│  Feature_vectors_classification_by_Adaboost.py
│  Feature_vectors_classification_by_Logistic_Regression.py
│  Feature_vectors_of_sticking_region.csv
│  structure.txt

## Models
### Model Used
The project utilizes the following machine learning models:

Logistic regression (linear)

Adaboost (ensemble model)

These two machine learning models—AdaBoost and Logistic Regression—are utilized to differentiate between true and false sticking breakout samples. We employ these models to evaluate whether an ensemble approach (AdaBoost) outperforms a single linear model (Logistic Regression) in handling binary classification tasks, as well as to identify the optimal model for breakout prediction.

## Results
The following is a summary of the testing results obtained for Adaboost model:

Best feature combination: [H, S, Rave, F_D, S_E, Edge_pnum]

Confusion matrix:

![image](https://github.com/user-attachments/assets/f52381a0-ddcb-4d93-9375-0e6c9f8f6b21)

Missing alarms: 0

False alarms: 22 (False alarm rate=13.9%)

Recall: 1.00

G-mean: 0.922

AUC: 0.93

AdaBoost outperforms the Logistic Regression model by combining weak classifiers, enabling it to fit complex decision boundaries and achieve better classification results on nonlinear datasets. In contrast, the Logistic Regression model is better suited for handling linear datasets. For a more detailed explanation and analysis, please refer to the accompanying paper.
