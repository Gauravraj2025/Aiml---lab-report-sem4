# Applied Machine Learning Lab Report

**University of Petroleum and Energy Studies (UPES)**
Dehradun, Uttarakhand

---

**Name:** Gaurav Raj
**SAP ID:** 590017701
**Batch:** 19
**Submitted To:** Dr. Sahinur Rahman Laskar
**Year:** 2025–2026

---

## Table of Contents

1. [Lab Experiment 1](#lab-experiment-1)
2. [Lab Experiment 2 – Predicting Housing Prices](#lab-experiment-2--predicting-housing-prices)
3. [Lab Experiment 3 – Predicting Stock Prices](#lab-experiment-3--predicting-stock-prices)
4. [Lab Experiment 4 – Customer Churn Prediction](#lab-experiment-4--customer-churn-prediction)
5. [Lab Experiment 5 – Spam Email Detection](#lab-experiment-5--spam-email-detection)
6. [Lab Experiment 6 – Credit Risk Assessment](#lab-experiment-6--credit-risk-assessment)
7. [Lab Experiment 7 – Anomaly Detection](#lab-experiment-7--anomaly-detection)
8. [Lab Experiment 8 – Student Performance Classification](#lab-experiment-8--student-performance-classification)
9. [Lab Experiment 9 – ECG Heartbeat Classification](#lab-experiment-9--ecg-heartbeat-classification)
10. [Lab Experiment 10 – Iris Flower Classification](#lab-experiment-10--iris-flower-classification)
11. [Lab Experiment 11 – Diabetes Prediction (Bagging Ensemble)](#lab-experiment-11--diabetes-prediction-bagging-ensemble)
12. [Lab Experiment 12 – House Price Prediction with Regularization](#lab-experiment-12--house-price-prediction-with-regularization)
13. [Mini Project 1 – Loan Prediction Pipeline](#mini-project-1--loan-prediction-pipeline)
14. [Mini Project 2 – Obesity Classification Pipeline](#mini-project-2--obesity-classification-pipeline)

---

## Lab Experiment 1

*(Content from Experiment 1 — see full document for details.)*

---

## Lab Experiment 2 – Predicting Housing Prices

**Problem:** Develop a regression model to predict house prices based on features like location, size, and amenities.

### Pipeline

#### 1. Problem Definition
The first step is understanding what we want to achieve. We must clearly define the goal, such as predicting house prices, student marks, or company profit. If the objective is not clear, we may choose the wrong approach. A well-defined problem gives direction to the entire project.

#### 2. Data Understanding
After defining the problem, we carefully study the dataset. We check the number of rows and columns, identify numerical and categorical features, and look for missing or unusual values. Since real-world data is often messy, understanding it properly helps avoid mistakes later.

#### 3. Data Preprocessing
Before using the data, we must prepare it. Machines understand only numbers, so categorical values like "Male" or city names must be converted into numerical form. We may also handle missing values and adjust the scale of features. Proper preprocessing ensures the model receives clean and usable data.

#### 4. Feature and Target Separation
Next, we divide the dataset into features (input variables) and the target (output variable). Features contain the information used for prediction, while the target is what we want to predict. This separation allows the model to learn the relationship between inputs and output.

#### 5. Train–Test Split
The data is then divided into training and testing sets. The model learns patterns from the training data and is evaluated on the testing data. This helps measure how well the model performs on unseen data and prevents memorization.

#### 6. Model Selection – Linear Regression
In this case, Linear Regression is chosen because the goal is to predict a continuous numerical value. It finds the best straight-line relationship between input features and the target variable, showing how each feature influences the prediction.

#### 7. Training and Prediction
During training, the model learns from the training data by adjusting its parameters to reduce error. After training, it is used to predict values for the test data. These predictions are compared with actual values.

#### 8. Model Evaluation
Finally, we evaluate the model using metrics like Mean Squared Error (MSE) and R² score. MSE measures the average prediction error, where a lower value is better. R² indicates how well the model explains the data, with values closer to 1 showing better performance.

---

## Lab Experiment 3 – Predicting Stock Prices

**Problem:** Develop a time series prediction model to forecast stock prices.

### Pipeline

#### 1. Problem Definition
Predicting future stock prices using historical data. Key decisions include whether you're forecasting the next day's closing price, the next week's average, or intraday movements — each choice changes how data is prepared and modeled.

#### 2. Data Understanding
Typical inputs include:
- Historical prices (open, high, low, close)
- Trading volume (how many shares were traded)
- Market indicators (moving averages, RSI, etc.)
- External factors (economic news, interest rates, global events)

#### 3. Preprocessing (Encoding)
- Handle missing values (fill gaps with averages or interpolation)
- Encode categorical data (e.g., "day of week" into numbers)
- Normalize or scale values (so features like trading volume don't overshadow price)
- Create lag features (yesterday's price, last week's average) to capture time dependencies

#### 4. Feature–Target Split
- **Features:** past prices, indicators, volumes, external signals
- **Target:** the price to forecast (e.g., tomorrow's closing price)

This split ensures the model knows what it's learning from and what it's trying to predict.

#### 5. Train–Test Split
- **Training set:** teaches the model
- **Test set:** evaluates how well the model generalizes

> For time series, data is **not** shuffled randomly — it is split chronologically. The model trains on older data and tests on newer data, mimicking real-world forecasting.

#### 6. Linear Regression (Model Selection)
Linear regression is a starting point. It assumes a straight-line relationship between inputs and outputs. For stock prices, this can capture basic trends but may struggle with complex patterns. More advanced models like ARIMA, LSTMs, or gradient boosting may be explored later.

#### 7. Training
The model learns by minimizing error between predicted and actual prices. In regression, this often means adjusting coefficients so the line of best fit captures the relationship between features and target.

#### 8. Prediction
Once trained, the model can forecast future prices. For example, given yesterday's price and trading volume, it predicts today's closing price.

#### 9. Evaluation
Accuracy is measured using:
- **Mean Absolute Error (MAE):** average difference between predicted and actual prices
- **Root Mean Squared Error (RMSE):** penalizes larger errors more heavily
- **R² Score:** shows how much variance in prices is explained by the model

> For stock data, evaluation is tricky because markets are noisy. A model might look good historically but fail in live trading.

---

## Lab Experiment 4 – Customer Churn Prediction

**Problem:** Develop a model to predict customer churn in a subscription-based business.

### Pipeline

#### 1. Problem Definition
Predict whether a customer will churn (cancel their subscription) based on their behavior and profile. This is a **classification problem** — the model outputs "churn" or "not churn."

#### 2. Data Understanding
Typical inputs include:
- Customer demographics (age, location, income level)
- Subscription details (plan type, tenure, billing cycle)
- Usage behavior (login frequency, time spent, features used)
- Support interactions (complaints, tickets raised, resolution time)
- Payment history (late payments, failed transactions)

#### 3. Preprocessing
- Missing values: filled with averages or dropped
- Categorical variables (e.g., "plan type", "region") converted into numbers
- Numerical features normalized for balance
- New features engineered (e.g., average logins per week, days since last activity)

#### 4. Feature–Target Split
- **Features:** all customer attributes and behaviors
- **Target:** churn status — `1` for churned, `0` for retained

#### 5. Train–Test Split
Dataset divided into training and testing sets to ensure the model generalizes to new, unseen customers.

#### 6. Model Selection
- **Logistic Regression:** easy to interpret, shows which factors are most influential
- **Decision Trees / Random Forests:** capture non-linear relationships
- **Gradient Boosting (XGBoost, LightGBM):** high accuracy, handles messy data well
- **Neural Networks:** powerful for very large datasets (harder to interpret)

#### 7. Training
The model studies the training data, adjusting internal parameters to minimize errors. Example: customers with low usage, frequent complaints, and short tenure are more likely to churn.

#### 8. Prediction
Models output probabilities (e.g., 85% chance of churning), enabling prioritization of high-risk customers for interventions.

#### 9. Evaluation
- **Accuracy:** general sense of correctness
- **Precision:** how many flagged churners actually churned
- **Recall:** how many actual churners were caught
- **F1 Score:** balance between precision and recall
- **ROC-AUC:** separates churners from non-churners

---

## Lab Experiment 5 – Spam Email Detection

**Problem:** Build a system that classifies emails as spam or ham (not spam) based on their content.

### Pipeline

#### 1. Problem Definition
Binary classification problem:
- Output = **Spam (1)** or **Ham (0)**

Goal: automatically filter unwanted emails and improve user experience.

#### 2. Data Understanding
Typical data includes:
- Email text (main feature)
- Label: Spam or Ham

Unlike other datasets, this focuses mainly on **text data**, not numerical features.

#### 3. Preprocessing (Text to Numbers)
Raw text cannot be directly used by ML models, so it is converted into numerical form.

Steps:
- Remove unnecessary characters (punctuation, symbols)
- Convert text to lowercase
- Remove stopwords (e.g., "the", "is")

**TF-IDF Vectorization:**
- Common words → low weight
- Important words (e.g., "free", "win", "offer") → high weight
- Each email becomes a numerical vector

#### 4. Feature–Target Split
- **Features (X):** TF-IDF vectors (numerical representation of email text)
- **Target (y):** `0` → Ham, `1` → Spam

```
Email Text → Spam/Ham
```

#### 5. Train–Test Split
- **Training Set:** used to train the model
- **Testing Set:** used to evaluate performance on unseen emails

#### 6. Model Selection
- **Naive Bayes** → very effective for text classification
- **Logistic Regression** → simple and interpretable
- **Support Vector Machine (SVM)** → powerful classifier
- **Decision Tree** → rule-based approach
- **Random Forest** → ensemble of multiple trees

#### 7. Training
Models learn patterns from email text and identify spam indicators. Example:
- Words like "free", "win", "offer" → spam
- Normal conversation words → ham

#### 8. Prediction
```
Input: "Congratulations! You won a free prize"
Output: Spam
```
Some models also provide probability (e.g., 90% chance of spam).

#### 9. Evaluation
- **Accuracy:** overall correctness of predictions
- **Precision:** out of predicted spam emails, how many are actually spam
- **Recall:** out of actual spam emails, how many were correctly detected
- **F1 Score:** balance between precision and recall

---

## Lab Experiment 6 – Credit Risk Assessment

**Problem:** Build a model to predict whether a customer is a credit risk based on their profile and financial information.

### Pipeline

#### 1. Problem Definition
Classification problem where the model assigns a category:
- Good risk
- Bad risk
- Average risk

Goal: assist financial institutions in identifying risky customers before granting credit.

#### 2. Data Understanding
Dataset: `credit_risk_assessment_500_samples.csv`

- **Features (X):** customer-related details such as income, credit history, and other attributes
- **Target (y):** `Credit_Risk` column indicating the risk level

Since the target is categorical (e.g., High, Medium, Low), it is converted into numerical form using `LabelEncoder`.

#### 3. Preprocessing
- **Encoding:** categorical labels converted into numeric values using `LabelEncoder`
- **Feature Scaling:** applied using `StandardScaler` to ensure all features are on a similar scale (important for Logistic Regression and SVM)

#### 4. Feature–Target Split
```
Customer Data → Credit Risk
```

#### 5. Train–Test Split
- Training set: used to train the model
- Testing set: used to evaluate performance on unseen data

#### 6. Model Selection

| Model | Description |
|---|---|
| Logistic Regression | Simple and interpretable; shows influence of features |
| Random Forest | Ensemble of decision trees; improves accuracy, reduces overfitting |
| XGBoost | Boosting algorithm; high performance on structured financial data |

#### 7. Training
Models learn patterns from customer data and identify relationships between features and risk levels. Example: customers with low income or poor credit history may be classified as high risk.

#### 8. Prediction
```
Input: Customer financial and demographic data
Output: Predicted risk level (e.g., High risk)
```

#### 9. Evaluation
- **Accuracy:** measures overall correctness
- **Precision:** reliability of predicted high-risk cases
- **Recall:** how many actual high-risk customers were correctly identified
- **F1 Score:** balance between precision and recall

Comparing metrics across Logistic Regression, Random Forest, and XGBoost determines which model is best suited for real-world credit risk analysis.

---

## Lab Experiment 7 – Anomaly Detection

**Problem:** Identify unusual or abnormal data points (anomalies) from a dataset.

### Pipeline

#### 1. Problem Definition
Anomalies are observations that significantly differ from the majority of the data. These can indicate:
- Fraudulent activities
- Data errors
- Unusual system behavior

Output:
- **Normal (0)**
- **Anomaly (1)**

#### 2. Data Understanding
Dataset components:
- **Features (X):** `Feature_1`, `Feature_2`
- **Target (y):** `0` → Normal, `1` → Anomaly

Most values lie within a normal range; a few values are significantly higher or different, indicating anomalies.

#### 3. Preprocessing
- Ensure data is clean and structured
- Features are numerical — no encoding required
- Feature scaling using `StandardScaler` may be applied

#### 4. Feature–Target Split
```
Data Pattern → Normal/Anomaly
```

#### 5. Train–Test Split
- Training set: used to learn patterns
- Testing set: used to evaluate detection performance

#### 6. Model Selection

**Isolation Forest**
- Randomly splits data into partitions
- Anomalies are isolated quickly because they are rare
- Efficient for large datasets

**Local Outlier Factor (LOF)**
- Compares each data point with its neighbors
- Detects points with significantly lower density than surrounding points

**One-Class SVM**
- Learns a boundary around normal data
- Points outside the boundary are classified as anomalies

#### 7. Training
Models learn the structure and distribution of normal data. Example:
- Values within 40–60 → normal
- Values above 90 → potential anomalies

#### 8. Prediction
```
Input: Feature values
Output: 0 → Normal | 1 → Anomaly
```

#### 9. Evaluation
- **Accuracy:** overall correctness
- **Precision:** how many detected anomalies are truly anomalies
- **Recall:** how many actual anomalies are detected
- **F1 Score:** balance between precision and recall

#### 10. Visualization
Results are visualized using scatter plots:
- Normal data points form clusters
- Anomalies appear as isolated or distant points

---

## Lab Experiment 8 – Student Performance Classification

**Problem:** Classify students into different performance levels based on academic and behavioral features.

### Pipeline

#### 1. Problem Definition
Multi-class classification problem:
- Output: Student performance category (e.g., Low, Medium, High)

Real-world application: identify weak students, enable personalized learning and academic improvement.

#### 2. Data Understanding
Dataset details:
- Total samples: 500 students
- Contains both numerical and categorical features

Input features: Participation (categorical), Internet Access (categorical), Previous Grade (categorical)

Target variable: `Performance_Level` (multi-class output)

#### 3. Preprocessing (Encoding & Scaling)

**Label Encoding:**
- Categorical columns converted to numeric using `LabelEncoder`
- Example: `Yes → 1`, `No → 0`

**Feature Scaling using `StandardScaler`:**
$$Z = \frac{X - \mu}{\sigma}$$

#### 4. Feature–Target Split
```
Student Data → Performance Level
```

#### 5. Train–Test Split
- Training Set: 80%
- Testing Set: 20%
- `stratify=y` used to ensure equal class distribution
- `random_state=42` ensures reproducibility

#### 6. Model Selection

| Model | Description |
|---|---|
| Decision Tree | Rule-based tree structure; easy to interpret; can overfit |
| Random Forest | Multiple decision trees; majority voting; reduces overfitting |
| Logistic Regression | Linear model; works well with simple relationships |
| K-Nearest Neighbors (KNN) | Classifies based on nearest data points |
| XGBoost | Advanced boosting; handles complex patterns with high accuracy |

#### 7. Training
Models learn the relationship between features and performance level. Example:
- High participation + good previous grade → High performance
- Low internet access → may affect performance negatively

#### 8. Prediction
```
Input: [Participation=High, Internet=Yes, Grade=A]
Output: High Performance
```

#### 9. Evaluation
$$Accuracy = \frac{Correct\ Predictions}{Total\ Predictions}$$

Metrics:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (shows correct vs. incorrect predictions)

---

## Lab Experiment 9 – ECG Heartbeat Classification

**Problem:** Classify ECG heartbeats as Normal (0) or Abnormal (1).

### Pipeline

#### 1. Problem Definition
Binary classification problem.
- Output: `0` → Normal heartbeat, `1` → Abnormal heartbeat

Real-world importance: used in heart disease detection, helps doctors identify abnormal heart conditions early.

#### 2. Data Understanding
- **Source:** MIT-BIH ECG dataset
- Each row represents one ECG sample
- **Features:** Multiple numerical values representing ECG signal points
- **Target:** Last column — `0` (Normal) or `1` (Abnormal)

#### 3. Preprocessing (Scaling & Label Conversion)

**Binary Conversion:**
```python
y = y.apply(lambda x: 0 if x == 0 else 1)
```
Converts multi-class labels into binary: `0` → Normal, others → Abnormal.

**Feature Scaling using `StandardScaler`:**
- ECG values may have different ranges
- Improves model performance
- Ensures equal importance of all features

#### 4. Feature–Target Split
```
ECG Signal → Normal/Abnormal
```

#### 5. Train–Test Split
- Training set: 80%
- Testing set: 20%

#### 6. Model Selection: Random Forest
- Works well with large datasets
- Handles complex patterns
- Reduces overfitting
- Uses majority voting across multiple decision trees

#### 7. Training
```python
rf.fit(X_train, y_train)
```
Model learns patterns in ECG signals and identifies differences between normal and abnormal heartbeats.

#### 8. Prediction
```python
y_pred = rf.predict(X_test)
```
```
Input: ECG signal values
Output: 1 (Abnormal)
```

---

## Lab Experiment 10 – Iris Flower Classification

**Problem:** Classify iris flowers into different species based on physical measurements.

### Pipeline

#### 1. Problem Definition
Multi-class classification:
- **Setosa**
- **Versicolor**
- **Virginica**

#### 2. Data Understanding
Loaded using `scikit-learn`. Contains:
- 150 samples (flowers)
- 4 input features: Sepal length, Sepal width, Petal length, Petal width
- Target variable: Species of flower (3 classes)

This dataset is clean, balanced, and widely used for ML learning.

#### 3. Preprocessing (Scaling)
- No missing values — no cleaning required
- Feature scaling applied using `StandardScaler`
- Ensures all features have equal importance and improves model performance

#### 4. Feature–Target Split
- **Features (X):** All 4 flower measurements
- **Target (y):** Flower species label

#### 5. Train–Test Split
- Training set: 80%
- Testing set: 20%
- Random state fixed to ensure reproducibility

#### 6. Model Selection

| Model | Strength |
|---|---|
| Logistic Regression | Simple and fast |
| Decision Tree | Easy to understand |
| Random Forest | More accurate (ensemble method) |

#### 7. Training
Models learn patterns such as:
- Relationship between petal size and species
- Differences between flower classes

#### 8. Prediction
```
Given measurements → predict species
```

---

## Lab Experiment 11 – Diabetes Prediction (Bagging Ensemble)

**Aim:** Predict whether a person has diabetes based on features such as blood pressure, skin thickness, age, etc., using the bagging ensemble technique. Also perform comparative analysis among the Bagging Classifier, Random Forest, and Decision Tree Classifier.

### Pipeline

#### Problem Definition
- **Goal:** Predict whether a person has diabetes (Binary: `0` or `1`) based on medical features such as Glucose levels, Blood Pressure, BMI, and Age.
- **Task:** Binary Classification
- **Ensemble Focus:** Implement the Bagging (Bootstrap Aggregating) technique to reduce variance and improve prediction stability.

#### Step 1: Understand the Dataset
Dataset: `dataset_diabetes_exp_11.csv`

- **Target variable (y):** `Outcome` (0: Non-diabetic, 1: Diabetic)
- **Input features (X):** 8 numerical features including:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age

#### Step 2: Data Preprocessing

**(a) Data Cleaning (Imputation):**
A critical step was identifying "hidden" missing values. Since values like `0` for Blood Pressure or BMI are biologically impossible, these were replaced with the **Median** of their respective columns. This prevents the model from being misled by 0-value outliers.

**(b) Feature Scaling:**
`StandardScaler` was applied to ensure all medical measurements (which have very different ranges, like Insulin vs. Age) are standardized.

#### Step 3: Train–Test Split
- Split Ratio: **80% Training, 20% Testing**
- `random_state=42` ensures consistent data partitioning for each model

#### Step 4: Choose the Ensemble & Classification Models

| Model | Description |
|---|---|
| Decision Tree Classifier | Baseline model (single tree) |
| Bagging Classifier | Ensemble using 10 estimators; Bootstrap Aggregating trains on random subsets with replacement |
| Random Forest Classifier | More robust ensemble using 100 estimators; adds feature-randomness to reduce model correlation |

#### Step 5: Train the Models
Models are fitted on the training data. The `BaggingClassifier` is built by wrapping the `DecisionTreeClassifier`, demonstrating how meta-estimators function in Scikit-Learn.

#### Step 6: Make Predictions
Ensemble models use **Majority Voting**. Each of the 10 trees (Bagging) or 100 trees (Random Forest) predicts a class, and the class with the most "votes" becomes the final prediction.

#### Step 7: Model Persistence
Final models saved using `pickle`:
- `decision_tree_classifier.pkl`
- `random_forest_classifier.pkl`
- `bagging_classifier.pkl`

> **Significance:** Preserves the entire "forest" of trees so future predictions can be made instantly without needing the original 768-row dataset or retraining time.

---

## Lab Experiment 12 – House Price Prediction with Regularization

**Aim:** Implement L1 (Lasso) and L2 (Ridge) regularization techniques on the Melbourne Housing dataset to predict house prices and perform a comparative analysis between both models.

### Pipeline

#### Problem Definition
- **Goal:** Predict house prices based on features like rooms, distance, area, etc.
- **Task:** Regression Problem (continuous output – price)
- **Regularization Focus:**
  - **Lasso (L1)** → Feature selection
  - **Ridge (L2)** → Coefficient shrinkage

To reduce overfitting and improve model performance.

#### Step 1: Understand the Dataset
Dataset: `dataset_Melbourne_housing_FULL_exp_12.csv`
- **Target variable (y):** Price
- **Input features (X):** Rooms, Distance, Landsize, BuildingArea, YearBuilt, etc.
- Dataset contains: **34,857 rows**, numerical + categorical features

#### Step 2: Data Preprocessing

**(a) Data Cleaning (Imputation):**
- Removed irrelevant columns: `Address`, `Date`
- Missing values handled:
  - Numerical → Mean
  - Categorical → Mode

**(b) Feature Encoding:**
- Categorical variables converted using **One-Hot Encoding** (`get_dummies`)
- Final dataset becomes fully numerical (799 features)

#### Step 3: Train–Test Split
- Split Ratio: **80% Training, 20% Testing**
- `random_state = 42` ensures consistency across runs

#### Step 4: Model Training

**Lasso Regression (L1)**
- Uses absolute penalty
- Can reduce some coefficients to zero
- Performs feature selection

**Ridge Regression (L2)**
- Uses squared penalty
- Reduces coefficient values
- Keeps all features

#### Step 5: Train the Models
```python
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
```

#### Step 6: Make Predictions
```python
lasso_model.predict(X_test)
ridge_model.predict(X_test)
```

#### Step 7: Model Persistence
Models saved using `pickle`:
- `lasso_model.pkl`
- `ridge_model.pkl`

---

## Mini Project 1 – Loan Prediction Pipeline

**Course:** Applied Machine Learning (CSAI2017P_5)
**Program:** B.Tech (Computer Science and Engineering)
**Semester:** 4
**Academic Session:** Jan–May 2026

### 1. Problem Definition

**Goal:** Predict loan amount using:
- Age (`person_age`)
- Gender (`person_gender`)
- Education Level (`person_education`)
- Income (`person_income`)
- Employment Experience (`person_emp_exp`)
- Home Ownership (`person_home_ownership`)
- Loan Intent (`loan_intent`)
- Interest Rate (`loan_int_rate`)
- Credit Score (`credit_score`)

**Task:** Regression problem (loan amount is a continuous numeric value)

**Target Variable (y):** `loan_amnt`

```
loan_amnt = f(age, gender, education, income, experience, credit_score, ...)
```

### 2. Data Understanding

Each row represents one loan applicant.

**Numerical Features:**
- `person_age`
- `person_income`
- `person_emp_exp`
- `loan_int_rate`
- `loan_percent_income`
- `credit_score`
- `cb_person_cred_hist_length`

**Categorical Features:**
- `person_gender`
- `person_education`
- `person_home_ownership`
- `loan_intent`
- `previous_loan_defaults_on_file`

**Key Observations:**
Loan amount depends mainly on income, credit score, employment experience, and loan purpose.

**EDA includes:**
- Loan amount distribution graph
- Average loan amount by education

### 3. Data Preprocessing

**(a) Remove Missing Values:**
```python
dropna()
```

**(b) Convert Categorical Data into Numbers:**
Categorical features converted using **One-Hot Encoding**. Examples:
- `person_gender_Male`
- `person_education_Bachelors`
- `loan_intent_Personal`

> ML models cannot understand text data.

**(c) Feature Scaling:**
`StandardScaler` applied to normalize values for models like Lasso, Ridge, SVR, and KNN.

### 4. Splitting Ratio
- Training Data: **80%**
- Testing Data: **20%**

```python
train_test_split(test_size=0.2)
```

### 5. Model Training

Regression models trained:
1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Support Vector Regressor (SVR)
8. K-Nearest Neighbors Regressor (KNN)

### 6. Test Model

```python
model.predict(X_test)
```

### 8. Conclusion
This project predicts loan amount using regression techniques.

**Key Points:**
- Ensemble models like Random Forest perform best
- Linear models give moderate performance
- Proper preprocessing improves accuracy

---

## Mini Project 2 – Obesity Classification Pipeline

### 1. Problem Definition

**Objective:** Build a machine learning model that can predict the obesity level of an individual based on their physical and demographic attributes.

- Obesity is a major health issue
- Early prediction helps in prevention
- Useful in healthcare analytics and fitness tracking

**Task:** Supervised Learning — Multi-Class Classification

**Target Variable (y):** `Label` (Obesity Level)

Possible Classes:
- Underweight
- Normal Weight
- Overweight
- Obese

```
Obesity Level = f(Age, Gender, Height, Weight, BMI)
```

### 2. Data Understanding

Each row represents one person. The dataset contains both numerical and categorical features.

**Numerical Features:**
- `Age` → Age of person
- `Height` → Height in meters
- `Weight` → Weight in kg
- `BMI` → Body Mass Index

**Categorical Features:**
- `Gender` → Male / Female

**Exploratory Data Analysis (EDA):**

| Visualization | Purpose |
|---|---|
| Count Plot (Target Distribution) | Shows number of samples per class; identifies class imbalance |
| Histogram (Age, BMI) | Shows data distribution; detects skewness |
| Scatter Plot (Height vs Weight) | Shows relationship between body features; visualizes clusters |
| Scatter Plot (Age vs BMI) | Shows how BMI varies with age |
| Box Plot (BMI vs Gender) | Detects outliers; compares distribution |
| Correlation Heatmap | Shows relationships between numerical features |

**Key Observations:**
- BMI is the most influential feature
- Weight strongly correlates with obesity level
- Height helps differentiate BMI values
- Gender has minor influence

### 3. Data Preprocessing

**(a) Data Cleaning:**
- Removed unnecessary column: `ID`
- Ensured no missing values

**(b) Encoding Categorical Data:**

Target Encoding:
- `Label` → converted using `LabelEncoder`

Feature Encoding:
- `Gender` → converted using One-Hot Encoding
  - `Male → 1`, `Female → 0`

**(c) Feature Scaling using `StandardScaler`:**
$$Z = \frac{X - \mu}{\sigma}$$

- Ensures all features have equal importance
- Improves performance of Logistic Regression and KNN

### 4. Data Splitting

```python
train_test_split(test_size=0.2, random_state=42)
```

- Training Set: **80%**
- Testing Set: **20%**

### 5. Model Training

| Model | Description |
|---|---|
| Logistic Regression | Linear model; works well for simple relationships |
| Decision Tree | Rule-based model; easy to interpret |
| Random Forest | Ensemble method; combines multiple decision trees; reduces overfitting |
| K-Nearest Neighbors (KNN) | Distance-based model; depends on nearest data points |
| Gaussian Naive Bayes | Probabilistic model; assumes feature independence; performs best on test data |

### 8. Model Saving

After selecting the best model:

```python
pickle.dump(model, file)
```

**Steps:**
1. Select best model
2. Save using `pickle`
3. Load when needed

**Why save the model?**
- Avoid retraining
- Use in real applications
- Deploy in web apps

### 9. Conclusion

- Built a multi-class classification system
- Used real-world features like BMI and weight
- Applied preprocessing for better performance
- Compared multiple ML models
- Evaluated using standard metrics
- **Random Forest performed best**

**Final Outcome:** Accurate obesity prediction system that can be used in healthcare and fitness apps.

---

*Report generated from: `gaurav_raj_17701_B19_ML_Report.pdf`*
*UPES, Dehradun — Academic Year 2025–2026*
<h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
