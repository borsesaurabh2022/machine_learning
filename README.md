# Machine Learning

This repository contains machine learning projects and notebooks worked through while studying core ML concepts, from data preprocessing to model training and evaluation.

---

## Ml_Handbook

A collection of notebooks following hands-on ML topics chapter by chapter. Each notebook covers a specific area of machine learning with real datasets and working code.

Notebooks in this folder:

- Cap1_HousingProject.ipynb
- Cap2_MinstClassification.ipynb
- Cap3_TrainingModels.ipynb
- cap4_SupportVectorMachine.ipynb
- chap8_DeepLearningNN.ipynb
- Cap14_ConvoNuralNetwork.ipynb
- proj1_MathStudentPerfAna.ipynb

---

## Chapter 1 - Housing Project

An end-to-end machine learning project using the California housing dataset to predict median house prices.

What we did:

- Downloaded and loaded the housing dataset from a remote source
- Explored the data using info, describe, and histograms to understand distributions
- Split the data into train and test sets using both random and stratified sampling based on income categories
- Visualized geographic data using latitude and longitude scatter plots
- Engineered new features like rooms per household, bedrooms per room, and population per household
- Handled missing values using the median strategy and built a preprocessing pipeline
- Encoded categorical features (ocean proximity) using one-hot encoding
- Built custom transformers and combined numerical and categorical pipelines using ColumnTransformer
- Trained three models: Linear Regression, Decision Tree, and Random Forest
- Evaluated models using RMSE and cross-validation to check for overfitting
- Fine-tuned the Random Forest model using GridSearchCV
- Analyzed feature importances to understand which attributes mattered most
- Evaluated the final model on the test set
