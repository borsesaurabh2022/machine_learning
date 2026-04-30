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

---

## Chapter 2 - MNIST Digit Classification

A classification project using the MNIST dataset to identify handwritten digits, covering binary and multiclass classification along with key evaluation techniques.

Brief steps followed during the learning phase:

- Loaded the MNIST dataset (70,000 images, 28x28 pixels each) using fetch_openml
- Explored the data structure and visualized sample digit images
- Split the data into 60,000 training and 10,000 test samples
- Built a binary classifier using SGDClassifier to detect the digit 5
- Measured accuracy using cross-validation and understood why accuracy alone is misleading for imbalanced datasets
- Computed a confusion matrix to see true vs predicted classifications
- Calculated precision, recall, and F1 score to properly evaluate the classifier
- Explored the precision-recall tradeoff by adjusting the decision threshold
- Plotted the ROC curve and computed the AUC score to compare classifiers
- Extended to multiclass classification using SVM with the OvO strategy and SGDClassifier
- Performed error analysis using a normalized confusion matrix to spot which digits the model confuses most
