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

To test the Adaboost model, run: python Feature_vectors_classification_by_Adaboost.py

To test the Logisitic regression model, run: python Feature_vectors_classification_by_Logistic_Regression.py

3. Once the script executes, it will initiate a loop that iterates over various feature combinations with a fixed dimension. You can modify the number of features and their corresponding combinations in the script to meet your specific requirements.

4. Note: We apologize for including some code written in Chinese; we are working to provide a fully English version in future updates.
