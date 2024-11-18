import matplotlib.pyplot as plt

# Dataset 1 RFILS data
durations_dataset_1 = [0.002, 0.002, 0.004, 0.005, 0.007, 0.008, 0.008, 0.009]
objective_dataset_1 = [8076.07, 7943.77, 7855.77, 7840.80, 7829.70, 7826.07, 7825.97, 7812.67]

# Dataset 2 RFILS data
durations_dataset_2 = [0.001, 0.003, 0.004, 0.005, 0.006, 0.008, 0.009, 0.011]
objective_dataset_2 = [305.37, 284.87, 279.97, 275.50, 272.67, 270.20, 269.47, 268.83]

# Dataset 4 RFILS data
durations_dataset_4 = [0.001, 0.002, 0.003, 0.004, 0.006, 0.006, 0.007, 0.008]
objective_dataset_4 = [33392.70, 33322.40, 33296.63, 33269.70, 33267.77, 33259.07, 33256.47, 33256.47]

# Dataset 7 RFILS data
durations_dataset_7 = [0.010, 0.017, 0.027, 0.038, 0.048, 0.054, 0.058, 0.066]
objective_dataset_7 = [277.53, 254.03, 241.17, 232.30, 229.70, 229.23, 228.87, 227.97]

# Dataset 9 RFILS data
durations_dataset_9 = [0.010, 0.016, 0.029, 0.042, 0.050, 0.060, 0.064, 0.069]
objective_dataset_9 = [547622.30, 547587.27, 547472.00, 547461.87, 547459.90, 547456.93, 547456.67, 547456.40]

# Dataset 10 RFILS data
durations_dataset_10 = [0.011, 0.018, 0.032, 0.059, 0.081, 0.091, 0.108, 0.120]
objective_dataset_10 = [605.90, 564.83, 532.43, 503.83, 490.93, 485.97, 481.23, 479.47]

# Plotting the graphs
plt.figure(figsize=(14, 10))

# Dataset 1
plt.subplot(2, 3, 1)
plt.plot(durations_dataset_1, objective_dataset_1, marker='o', linestyle='-', color='blue')
plt.scatter(durations_dataset_1[5], objective_dataset_1[5], color='red', s=100, label='8000 plateau moves')
plt.title('Dataset 1')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Dataset 2
plt.subplot(2, 3, 2)
plt.plot(durations_dataset_2, objective_dataset_2, marker='o', linestyle='-', color='green')
plt.scatter(durations_dataset_2[5], objective_dataset_2[5], color='red', s=100, label='8000 plateau moves')
plt.title('Dataset 2')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Dataset 4
plt.subplot(2, 3, 3)
plt.plot(durations_dataset_4, objective_dataset_4, marker='o', linestyle='-', color='pink')
plt.scatter(durations_dataset_4[5], objective_dataset_4[5], color='red', s=100, label='8000 plateau moves')
plt.title('Dataset 4')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Dataset 7
plt.subplot(2, 3, 4)
plt.plot(durations_dataset_7, objective_dataset_7, marker='o', linestyle='-', color='purple')
plt.scatter(durations_dataset_7[5], objective_dataset_7[5], color='red', s=100, label='32000 plateau moves')
plt.title('Dataset 7')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Dataset 9
plt.subplot(2, 3, 5)
plt.plot(durations_dataset_9, objective_dataset_9, marker='o', linestyle='-', color='orange')
plt.scatter(durations_dataset_9[5], objective_dataset_9[5], color='red', s=100, label='32000 plateau moves')
plt.title('Dataset 9')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Dataset 10
plt.subplot(2, 3, 6)
plt.plot(durations_dataset_10, objective_dataset_10, marker='o', linestyle='-', color='cyan')
plt.scatter(durations_dataset_10[5], objective_dataset_10[5], color='red', s=100, label='32000 plateau moves')
plt.title('Dataset 10')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Adjust the layout
plt.tight_layout(pad=3.0)
plt.show()