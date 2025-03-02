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
- GPU Training Dependencies: PyTorch (with a version compatible with your CUDA and cuDNN installations).

### Steps to Install
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name
