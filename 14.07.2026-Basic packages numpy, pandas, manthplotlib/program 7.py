import pandas as pd

data = {
    "Name": ["Arun", "Priya", "Kavin", "Meena"],
    "Age": [20, 21, 19, 22],
    "Marks": [85, 92, 78, 88]
}

df = pd.DataFrame(data)

print(df)
