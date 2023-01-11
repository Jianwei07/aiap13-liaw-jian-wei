import sqlite3
import urllib.request
import pandas as pd

# Download the database file
urllib.request.urlretrieve("https://techassessment.blob.core.windows.net/aiap13-assessment-data/failure.db")

# Connect to the database
conn = sqlite3.connect('failure.db')

# Create a cursor
conn = sqlite3.connect('failure.db')
cursor = conn.cursor()

# Get all the table names
cursor.execute("SELECT name from sqlite_master WHERE type='table';")
table_names = cursor.fetchall()

# Iterate over the tables and print the data
for table in table_names:
    cursor.execute(f"SELECT * FROM {table[0]}")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

# Close the cursor and connection
cursor.close()
conn.close()
