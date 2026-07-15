import matplotlib.pyplot as plt

study_hours = [1, 2, 3, 4, 5, 6]
marks = [45, 50, 60, 70, 80, 90]

plt.scatter(study_hours, marks)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.show()
