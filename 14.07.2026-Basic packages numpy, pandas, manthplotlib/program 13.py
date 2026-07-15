import matplotlib.pyplot as plt

subjects = ["Python", "Java", "C++", "JavaScript"]
students = [35, 25, 20, 20]

plt.pie(students, labels=subjects, autopct="%1.1f%%")

plt.title("Programming Language Preference")

plt.show()
