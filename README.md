aiap13-liaw-jian-wei-026z

# **AIAP13 Technical Assessment**

## Name: Liaw Jian Wei
## Email: liawjianwei@outlook.com

This submission consists of the following documents
    1. .venv - Virtual enviroment created to host this project using vscode 
        a. to download the necessary extensions *i.e., pandas, seaborn, sqlite3*
        b. better control over file paths and location

    2. Task 1 - Exploratory Data Analysis (EDA) 
    _eda.ipynb_, Exploratory Data Analysis using jupyter notebook
        a. given the failure.db file, the eda.ipynb was coded in a way to read the file using *sqlite3*
        _data = pd.read_sql_ uses the pandas library to read data from a SQL table called "Failure" using a SQL query "SELECT * FROM Failure"
         and store the returned data in a variable called "data". 

        b. **print(), head,describe and infor** were use to show some data for counterchecking purposes

        c. _make_group_ method was used to perform aggregate function on the data to identify the different models and finding the mean of the numeric column
        a bar chart was created to show this data plotted "Models vs Failures"

        d. line chart was made to plot the mean of rpm vs temperature 
        
        e. _conn = sqlite3.connect("C:\\Users\\Jianw\\Documents\\GitHub\\aiap13-liaw-jian-wei-026z\\data aiap13\\failure.db")_ 
        a long pathway was keyed in because error kepy appear for data\failure.db even after editing the setting.json
        
###Conclusion
EDA as stated in the name, derives the purpose the analysis the data and present it in visuals using visualization tools. 
More visualizations can be done through dash.app. and plotly. 

    3. Task 2: End-to-end Machine Learning Pipeline (MLP)

        a. an exportcsv.ipynb was created to export failure.db as csv file to ease the analysis process

        b. using pandas, sklearn matrix and parameters, the csv data was processed. The data were split and trained into different test sets using the *n_train, n_test* parameters
    
        c. the model was trained using a random forest classifier, 






