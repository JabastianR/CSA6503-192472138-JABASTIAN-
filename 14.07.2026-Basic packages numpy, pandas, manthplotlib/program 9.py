import pandas as pd

data = {
    "Name": ["Arun", "Priya", "Kavin"],
    "Marks": [85, 92, 70]
}

df = pd.DataFrame(data)

df["Result"] = ["Pass" if mark >= 50 else "Fail" for mark in df["Marks"]]

print(df)
