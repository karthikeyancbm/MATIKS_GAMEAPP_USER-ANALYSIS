# MATIKS_GAMEAPP_USER-ANALYSIS

# Problem Statement:

  This project is about Game app Exploratory Data Analysis of its users and other parameters and prediction of revenue and user subscription category.

# Approach:

**1.Data Wrangling:**

It involves the following steps:

  * Data Cleaning
  * Data Transforamtion
  * Data Enrichment

**Data Cleaning:**

  * Handle missing values with mean/median/mode.
  * Treat Outliers using IQR
  * Identify Skewness in the dataset and treat skewness with appropriate data transformations,
    such as log transformation.
  * Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding,
    or ordinal encoding, based on their nature and relationship with the target variable.

**Data transformation:**

  *  Changing the structure or format of data, which can include normalizing data, scaling features, or encoding categorical variables.

**Exploratory Data Analysis:**

  * Analyse the past data to identify patterns, trends and explore it through visualization vide different plots
    using matplotlib and seaborn.

**Feature Engineering:**

  * Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data.

**Model Building and Evaluation:**

* Split the dataset into training and testing/validation sets.
* Train the different regression models and Classification models  and evaluate the result with suitable metrics such as MAE - Mean Absolute Error, MSE - Mean Squared Error and
  RMSE - Root Mean Squared Error for Regression Model and accuracy_score,precision_score,recall_score,f1-score for Classification models.

**Model Deployment using Streamlit:**
* Develop interactive GUI using streamlit.
* create an input field where the user can enter each column value except target variable.

