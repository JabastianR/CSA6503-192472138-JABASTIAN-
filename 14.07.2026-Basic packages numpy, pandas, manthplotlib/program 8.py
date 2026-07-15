import pandas as pd

data = {
    "Name": ["Arun", "Priya", "Kavin", "Meena"],
    "Marks": [85, 92, 65, 88]
}

df = pd.DataFrame(data)

high_marks = df[df["Marks"] > 80]

print("Students who scored above 80:")
print(high_marks)
