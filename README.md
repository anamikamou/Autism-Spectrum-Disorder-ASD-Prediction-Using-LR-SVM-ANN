### Autism-Spectrum-Disorder-ASD-Prediction-Using-LR-SVM-ANN

***Project Overview***
This project applies machine learning techniques to predict Autism Spectrum Disorder (ASD) using screening questionnaire data and demographic attributes. The objective is to explore how different classification algorithms perform in identifying ASD cases and to compare their predictive accuracy.

***Dataset: Exploratory Data Analysis(EDA)***
The dataset contains screening and demographic information used to identify ASD traits. 
Column Type	       Examples
Binary -->	       A1–A10, jundice, austim, used_app_before
Numeric-->	       age, result
Categorical-->	   gender, ethnicity, country, relation
Target Variable--> Class/ASD

1
A1_Score – A10_Scores: These 10 questions are involved in the core screening test. The ASD screening test itself is based on these questions. Models usually learn strong patterns from them. Many 1s Higher chance of ASD, Mostly 0s Lower chance. These are the most influential features. 
2
If the input have A1–A10 features, the ‘result’ feature becomes redundant because the model can calculate it automatically. So removing the feature ‘result’ is convenient.
3
Moderately Important Features: Family Autism History, Jaundice at Birth, Age. These features may influence ASD probability but not as strongly as the questionnaire scores.
4
Low Importance Features: gender, ethnicity, country, relation, used_app_before
These features usually contribute very little to prediction.
5
The feature ‘age_desc’ has only one value for all rows. It should be removed because it provides no useful information.
6
The most important factor in choosing between regression and classification is the target variable. Your target column is: Class/ASD, Values: YES/NO, this means the model must assign a category, not predict a number. This is a decision problem, not a numerical prediction. Therefore: Predicting YES or NO → Classification problem. So, by definition, this dataset fits classification. Regression models usually work best with continuous numerical variables.
Why These 3 Make a Good Comparison

***Data Preprocessing:***
The following preprocessing steps were performed:
1.	Removed an extreme outlier in the age column (value = 383)
During preprocessing, unrealistic values were identified in the age attribute (e.g., age = 383). Since such values represent data entry errors and may negatively influence model performance, these outliers were removed to maintain data quality. 
2.	Handled missing values
Now there were two null values in age attribute. so, as the dataset was not very large, the null values were filled up by median age number because age values are not perfectly normally distributed. This median age value ensured data robustness again skewness (pulled by extremes). As ethnicity and relation has missing values, those has been replaced with ‘others’.
3.	Encoded categorical variables into numerical format
After cleaning the age column, the next step is to Labelencode() categorical variables(gender, jaundice, austim, used_app_before, Class/ASD) so that machine learning algorithms (Logistic Regression, SVM, ANN) can understand them. ML models cannot work with text values, so we convert them to numbers. One-hot encoding( get_dummies() ) to convert multiple catagories into binary numbers.
4.	Split the dataset into training and testing sets
Train/Test method to split the data set into two sets: a training set and a testing set where 80% data was used to train and 20% was used to test the models.
5.	Applied feature scaling using StandardScaler()
As the features has dependency for distance calculations and gradient updates, it is important to scale data all the datas into new values to adjust features that are easier to compare. Especially, age value is large, so if the feature is not scaled, the model will be largely biased to the age feature.

***Machine Learning Models***
The most important factor in choosing between regression and classification is the target variable. The target column is: Class/ASD, Values: YES/NO, this means the model must assign a category, not predict a number. This is a decision problem, not a numerical prediction. Therefore: Predicting YES or NO → Classification problem. So, by definition, this dataset fits classification. Regression models usually work best with continuous numerical variables.
Three classification algorithms were implemented using the scikit-learn library:
1.	Logistic Regression
2.	Support Vector Machine (SVM)
3.	Artificial Neural Network (ANN)
   
***Model Evaluation***
The models were evaluated using standard classification metrics:
1.	Accuracy
2.	Confusion Matrix
3.	Precision, Recall, F1- score, Support

***Results***
The performance of the models was compared based on accuracy.

Model                               Accuracy
Logistic Regression:	            97%
Support Vector Machine:	          100%
Artificial Neural Network:	      92%

Here support vector machine model provides 100% accuracy. Because of strong relationship of questionaries features (A1 to A10) with output feature Class/ASD, this model provides strong accuracy. Moreover, these screening scores records demonstrate behavioral indicators corelates with Autism Spectrum Disorder, which enables the SVM model to achieve perfect accuracy on the testing data. Although the svm model achieved a perfect accuracy, further procedures such as cross-validation or testing on very large dataset would be necessary to confirm the generalization ability of the svm model. Additionally, features which are highly associated with output and influence the output were shown by feature importance function. 

***Conclusion***
This project demonstrates how machine learning models can be applied to classify Autism Spectrum Disorder using screening questionnaire data. The comparison of Logistic Regression, SVM, and ANN provides insight into how different algorithms perform on the same dataset and highlights the importance of proper preprocessing and evaluation in classification tasks.

***Technologies Used***
•	Python
•	Pandas
•	NumPy
•	scikit-learn
•	Jupyter Notebook

Dataset link:
https://www.kaggle.com/datasets/faizunnabi/autism-screening

***How to Run the Project***
1.	Clone the repository
https://github.com/anamikamou/Autism-Spectrum-Disorder-ASD-Prediction-Using-LR-SVM-ANN
cd Credit_Card_Fraud_Detection_ML_Project
2.	Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
3.	Run the notebook
Open Jupyter Notebook or VS Code:
jupyter notebook asd_final.ipynb




