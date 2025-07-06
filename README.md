⚽ Striker Performance Analysis & Classification
A complete data science project analyzing striker performance based on various metrics. The project includes data preprocessing, EDA, statistical testing, clustering (KMeans), feature engineering, and classification using Logistic Regression.

📊 Problem Statement
Goal: Analyze and classify strikers into 3 categories — Regular Strikers, Good Strikers, and Best Strikers — based on their performance metrics.

🧪 Tools & Technologies Used
  Python (Pandas, Numpy, Matplotlib, Seaborn)
  
  scikit-learn (StandardScaler, KMeans, LogisticRegression)
  
  statsmodels (OLS Regression)
  
  SciPy (ANOVA, Pearson correlation)
  
  Visual Studio

   Data Preprocessing
Handled missing values using median imputation

Label encoded Footedness and Marital Status

One-hot encoded Nationality

Created a new feature: Total Contribution Score

📊 Statistical Analysis
Normality Test: Shapiro-Wilk

ANOVA: Tested if consistency rates differ across nationalities

Pearson Correlation: Checked if Hold-up Play relates to Consistency

Linear Regression: Determined how much Hold-up Play influences Consistency

🤖 Machine Learning
Applied KMeans Clustering to group strikers into:

Regular Strikers

Good Strikers

Best Strikers

Trained a Logistic Regression Classifier to predict striker type

Achieved ~93–97% accuracy

Evaluated using:

Confusion Matrix

Classification Report (Precision, Recall, F1)

You can input any new player stats like this:

python
Copy
Edit
new_player = {
    'Goals Scored': 20,
    'Assists': 10,
    'Shots on Target': 35,
    ...
    'Nationality_France': 1,
    ...
}
And the model will return:

python
Copy
Edit
Predicted Striker Type: Best Striker


📜 License
This project is for educational use only. All rights to the dataset belong to the original authors.
