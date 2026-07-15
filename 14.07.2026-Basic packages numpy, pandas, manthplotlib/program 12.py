import matplotlib.pyplot as plt

students = ["Arun", "Priya", "Kavin", "Meena"]
marks = [85, 92, 78, 88]

plt.bar(students, marks)

plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks")

plt.show()
