import pandas as pd

try:
    df = pd.read_csv("students.csv")

    print("CSV File Data:")
    print(df)

except FileNotFoundError:
    print("The students.csv file was not found.")
