import matplotlib.pyplot as plt

# Data for SA accepting equal moves (Dataset 1, 7, 13)
durations_accept_equal_1 = [0.007, 0.015, 0.020, 0.029, 0.039]
objective_accept_equal_1 = [8155.87, 5055.13, 4944.23, 4930.87, 4866.53]

durations_accept_equal_7 = [0.054, 0.085, 0.143, 0.186, 0.225]
objective_accept_equal_7 = [304.73, 213.47, 196.60, 190.03, 186.03]

durations_accept_equal_13 = [0.371, 0.565, 0.795, 1.058, 1.423]
objective_accept_equal_13 = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]

# Data for SA not accepting equal moves with prob. 1 (Dataset 1, 7, 13)
durations_no_equal_1 = [0.007, 0.015, 0.021, 0.027, 0.039]
objective_no_equal_1 = [8216.10, 5081.23, 4975.03, 4959.67, 4921.97]

durations_no_equal_7 = [0.055, 0.087, 0.144, 0.189, 0.228]
objective_no_equal_7 = [305.03, 214.90, 197.87, 192.67, 190.13]

durations_no_equal_13 = [0.389, 0.567, 0.798, 1.090, 1.395]
objective_no_equal_13 = [2648.47, 2171.93, 2142.43, 2122.47, 2104.93]

# Plotting the graphs
plt.figure(figsize=(18, 5))

# Plot for Dataset 1
plt.subplot(1, 3, 1)
plt.plot(durations_accept_equal_1, objective_accept_equal_1, marker='o', linestyle='-', color='red', label='Accept Equal Moves')
plt.plot(durations_no_equal_1, objective_no_equal_1, marker='o', linestyle='-', color='green', label='No Equal Moves (prob. 1)')
plt.title('Dataset 1: SA Accept vs No Equal Moves')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()
plt.grid(True)

# Plot for Dataset 7
plt.subplot(1, 3, 2)
plt.plot(durations_accept_equal_7, objective_accept_equal_7, marker='o', linestyle='-', color='red', label='Accept Equal Moves')
plt.plot(durations_no_equal_7, objective_no_equal_7, marker='o', linestyle='-', color='green', label='No Equal Moves (prob. 1)')
plt.title('Dataset 7: SA Accept vs No Equal Moves')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()
plt.grid(True)

# Plot for Dataset 13
plt.subplot(1, 3, 3)
plt.plot(durations_accept_equal_13, objective_accept_equal_13, marker='o', linestyle='-', color='red', label='Accept Equal Moves')
plt.plot(durations_no_equal_13, objective_no_equal_13, marker='o', linestyle='-', color='green', label='No Equal Moves (prob. 1)')
plt.title('Dataset 13: SA Accept vs No Equal Moves')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()
plt.grid(True)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()