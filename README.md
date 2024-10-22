# **AIAP13 Technical Assessment** 
_aiap13-liaw-jian-wei-026z_

## Name: Liaw Jian Wei
## Email: liawjianwei@outlook.com

This submission consists of the following documents
 1. .venv - Virtual enviroment created to host this project using vscode
    a. to download the necessary extensions *i.e., pandas, seaborn, sqlite3*
    b. better control over file paths and location
        
## 2. Task 1 - Exploratory Data Analysis (EDA) 
_eda.ipynb_, Exploratory Data Analysis using jupyter notebook

### OUTLINE:  PROGRAM OUTLINE
              (1) IMPORT LIBRARIES AND INITIALISATION. (SQLITE3)
              (2) READ IN DATA AND VERIFICATION. (PANDAS, PRINT)
              (3) DATA CLEANING AND TRANSFORMATION
              (4) DATA VISUALIZATION (Linechart)
              (5) CONCLUSION AND FOLLOWUP TO TASK 2

a. given the failure.db file, the eda.ipynb was coded in a way to read the file using *sqlite3*
_data = pd.read_sql_ uses the pandas library to read data from a SQL table called "Failure" 
using a SQL query "SELECT * FROM Failure" and store the returned data in a variable called "data". 
       
b. **print(), head,describe and infor** were use to show some data for counterchecking purposes prior to analyse the data. 

c. _make_group_ method was used to perform aggregate function on the data to identify the 
different models and finding the mean of the numeric column a bar chart was created to 
show this data plotted "Models vs Failures"

d. line chart was made to plot the mean of rpm vs temperature 
        
e. _conn = sqlite3.connect("C:\\Users\\Jianw\\Documents\\GitHub\\aiap13-liaw-jian-wei-026z\\data aiap13\\failure.db")_ 
a long pathway was keyed in because error kepy appear for data\failure.db even after editing the setting.json

f. To ease identifying the failures againist the model, a code was wrriten to total up the failures and export
as a csv file. 
_data = pd.read_sql_query("SELECT * FROM Failure", conn)_
_data["Failure"] = data[["Failure A","Failure B","Failure C","Failure D","Failure E"]].sum(axis=1)_

### Conclusion
EDA as stated in the name, derives the purpose of analysing the data and present it in visuals using data visualization tools. 
More visualizations can be done through dash.app. and plotly. 

## 3. Task 2: End-to-end Machine Learning Pipeline (MLP)
Designing and creating a machine learning pipeline in Python to predict the occurrence of car failure 
using the provided dataset for an automotive company would involve several steps, including data ingestion, 
data preprocessing, feature engineering, model selection, model evaluation and visualization. 

    a. an exportcsv.ipynb was created to export failure.db as csv file to ease the analysis process

    b. using pandas, sklearn matrix and parameters, the csv data was processed. 
    The data were split and trained into different test sets using the *n_train, n_test* parameters
    
    c. the model was trained using a random forest classifier, Decision Tree and Logistic Regression to train the model

    d. Confusion Matrix 2x2 layout was present for a binary classification type as it is either a yes or a no 

    e. a bar chart was exported as visualization for this code 

_Unable to identify mistake to fix run.sh after trying different ways to connect MLP.py pathway to run.sh. 
MLP.py individually is able to produce results._

### Personal Thoughts
This assessment was an eye-opening one from the usual hackerrank/hackerearth test. This requires the programmer to sort 
and test the data which refreshed my memory and taught me new things. I do hope that I am given the opportunity to learn 
and work with AIAP to further add value to any industry through digitilization. 

    
    






