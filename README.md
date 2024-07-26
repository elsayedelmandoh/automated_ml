# Automated Machine Learning Model

This project is an end-to-end automated machine learning (AutoML) model that streamlines the process of data acquisition, preprocessing, building, comparison, evaluating, testing, visualizing and saving machine learning models for both classification and regression tasks. It is designed to help users with minimal data science expertise quickly go from raw data to a well-performing machine learning model.

## Table of Contents

- [Overview](#overview)
- [Hosting](#hosting)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributing](#contributing)
- [Author](#author)

## Overview

The AutoML model follows a structured workflow and provides a user-friendly interface using Streamlit. It includes the following steps:

1. **Data Acquisition**: Load your dataset and define the target variable.

2. **Data Exploration and Analysis (EDA)**: Perform exploratory data analysis to understand the dataset's characteristics, visualize distributions, and analyze relationships between features.

3. **Data Preparation**: Preprocess the data by handling missing values, dropping duplicates, encoding categorical features, and scaling numerical features.

4. **Model Creation**: Select from a variety of machine learning models for both classification and regression tasks. You can choose specific models or apply all models available.

5. **Model Comparison**: Train and evaluate the selected models using training and validation data. The best-performing model is identified based on metrics such as accuracy, recall, precision (for classification), or mean absolute error, mean squared error, R-squared (for regression).

6. **Model Testing**: Test the best model on a separate test dataset and generate a testing report.

7. **Model Visualization**: Visualize the performance of the best model on training, validation, and test datasets using bar plots for various metrics.


## Hosting

https://automatedml-i6676wgyc7bvbz3tqj5gkh.streamlit.app/


## Requirements

- Python 3.6+
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/elsayedelmandoh/automated_ml.git

2. Install the required dependencies using command pip install -r requirements.txt
3. Run this command streamlit run main.py


## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional content to contribute, feel free to open issues, submit pull requests, or provide feedback. 

[![GitHub watchers](https://img.shields.io/github/watchers/elsayedelmandoh/automated_ml.svg?style=social&label=Watch)](https://GitHub.com/elsayedelmandoh/automated_ml/watchers/?WT.mc_id=academic-105485-koreyst)
[![GitHub forks](https://img.shields.io/github/forks/elsayedelmandoh/automated_ml.svg?style=social&label=Fork)](https://GitHub.com/elsayedelmandoh/automated_ml/network/?WT.mc_id=academic-105485-koreyst)
[![GitHub stars](https://img.shields.io/github/stars/elsayedelmandoh/automated_ml.svg?style=social&label=Star)](https://GitHub.com/elsayedelmandoh/automated_ml/stargazers/?WT.mc_id=academic-105485-koreyst)

## Author

This repository is maintained by Elsayed Elmandoh, an AI Engineer. You can connect with Elsayed on [LinkedIn and Twitter/X](https://linktr.ee/elsayedelmandoh) for updates and discussions related to Machine learning, deep learning and NLP.

Happy coding!
