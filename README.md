### Heart Disease Prediction

This repository contains a Jupyter Notebook for predicting heart disease based on patient data. The project utilizes a Random Forest Classifier to build a predictive model, offering insights into the factors that are most influential in determining the presence of heart disease.

### Files

-   `Heart_Disease_Prediction (1).ipynb`: The main Jupyter Notebook containing all the code for data processing, model training, and prediction.
-   `heart-rf.pkl`: The trained Random Forest Classifier model saved as a pickle file.
-   `heart-scaler.pkl`: The `StandardScaler` used to scale the data, saved as a pickle file.
-   `heart_user_template.csv`: A template CSV file for users to input their own data for prediction.
-   `heart_dataset.csv`: A sample dataset for demonstration purposes.

### Project Description

The goal of this project is to predict the presence of heart disease in patients. The approach involves:

1.  **Data Loading and Cleaning**: The dataset is loaded from Kaggle, and initial data exploration is performed to identify and handle missing values.
2.  **Exploratory Data Analysis (EDA)**: Visualizations like histograms and correlation matrices are used to understand the distribution of numerical features and their relationships.
3.  **Data Preprocessing**: Categorical features are converted into a numerical format using one-hot encoding, and the data is scaled to prepare it for model training.
4.  **Model Training**: A Random Forest Classifier model is trained on a portion of the dataset.
5.  **Model Evaluation**: The model's performance is evaluated using metrics such as accuracy, a classification report, and a confusion matrix.
6.  **Feature Importance**: The most important features for the model's predictions are identified and visualized.
7.  **Prediction on New Data**: The trained model and scaler are saved to make predictions on new, unseen data provided by a user.

### Setup and Usage

1.  Clone this repository to your local machine.
2.  Open the `Heart_Disease_Prediction (1).ipynb` file in a Jupyter environment (like Google Colab).
3.  Execute the cells in the notebook sequentially.
4.  To make a prediction on new data, you can upload a CSV file with the same format as `heart_user_template.csv` when prompted.

---

## Presentation Slides (PPT)

### Slide 1: Title Slide

**Title:** Heart Disease Prediction using Machine Learning  
**Subtitle:** A Random Forest Classifier Approach  
**Presenter:** [Your Name]

---

### Slide 2: Project Overview

**Heading:** What is this project about?  
**Body:**
-   **Goal:** To build a machine learning model that predicts the presence of heart disease in patients.
-   **Dataset:** We use a publicly available heart disease dataset from Kaggle.
-   **Methodology:** The project follows a standard machine learning workflow, from data cleaning and exploration to model training and evaluation.

---

### Slide 3: Data Loading and Preprocessing

**Heading:** Preparing the Data  
**Body:**
-   **Source:** The data is imported directly from Kaggle.
-   **Missing Values:** We handled missing values by filling them. Numerical columns were filled with the mean of their respective columns.
-   **Categorical Features:** We used one-hot encoding to convert categorical data (e.g., 'sex', 'cp') into a numerical format that the model can understand. This is a crucial step to avoid misinterpreting these features as ordinal.

---

### Slide 4: Exploratory Data Analysis (EDA)

**Heading:** Understanding the Data  
**Body:**
-   **Histograms:** We visualized the distributions of numerical features like `age`, `chol`, and `thalch` using histograms. This helped us understand the data's spread and potential outliers.
-   **Correlation Matrix:** A heatmap was used to visualize the correlation between numerical features. This is vital for identifying multicollinearity and understanding which features might be strongly related to each other or the target variable.

---

### Slide 5: Model Training

**Heading:** Building the Predictive Model  
**Body:**
-   **Model:** We chose the **Random Forest Classifier**, a powerful ensemble learning method, for this task. It's known for its high accuracy and ability to handle complex datasets.
-   **Data Split:** The data was split into a training set (80%) and a testing set (20%) to ensure the model is evaluated on data it has not seen before.
-   **Scaling:** We used `StandardScaler` to scale the features. This is important for many machine learning algorithms, including Logistic Regression, to perform well by standardizing the range of independent variables.

---

### Slide 6: Model Evaluation

**Heading:** How well does the model perform?  
**Body:**
-   **Accuracy Score:** The model achieved an accuracy of approximately **88.59%**.
-   **Classification Report:** A detailed report provides metrics like **precision**, **recall**, and **f1-score** for both classes (presence or absence of heart disease).
    -   Precision: The proportion of positive identifications that were actually correct.
    -   Recall: The proportion of actual positives that were correctly identified.
-   **Confusion Matrix:** A heatmap of the confusion matrix shows the number of true positives, true negatives, false positives, and false negatives.

---

### Slide 7: Feature Importance

**Heading:** What are the most important factors?  
**Body:**
-   A bar chart visualizes the importance of each feature in the model's prediction.
-   Features like `ca` (number of major vessels), `thal` (thalassemia), `oldpeak` (ST depression), and `cp` (chest pain type) are highlighted as the most influential in determining heart disease. This provides valuable insights into which clinical measurements are most predictive.

---
