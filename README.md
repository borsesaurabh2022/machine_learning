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

---

## Chapter 3 - Training Models

A deep dive into how machine learning models are actually trained under the hood, covering linear regression, gradient descent variants, regularization techniques, and logistic regression.

What we did:

- Generated synthetic data and fit a Linear Regression model using the Normal Equation
- Understood how the Normal Equation computes optimal parameters directly without iterating
- Implemented Batch Gradient Descent from scratch and experimented with different learning rates
- Implemented Stochastic Gradient Descent and Mini-batch Gradient Descent and compared their behavior
- Plotted learning curves to diagnose underfitting and overfitting
- Used Polynomial Regression to fit non-linear data and observed how model complexity affects the bias-variance tradeoff
- Applied Ridge Regression (L2) to penalize large weights and reduce overfitting
- Applied Lasso Regression (L1) and observed how it drives less useful feature weights to zero
- Used Elastic Net as a balance between Ridge and Lasso regularization
- Implemented Early Stopping to halt training when validation error stops improving
- Built a Logistic Regression model for binary classification and understood the sigmoid function
- Extended to Softmax Regression for multiclass classification using the cross-entropy loss
