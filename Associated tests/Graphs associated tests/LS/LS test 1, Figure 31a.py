import matplotlib.pyplot as plt

# Data for BILS
durations_bils = [0.009, 0.014, 0.020, 0.028, 0.037]
objective_bils = [15670.17, 12978.07, 10343.97, 9649.67, 9510.87]

# Data for RFILS
durations_rfils = [0.007, 0.011, 0.025, 0.035, 0.043]
objective_rfils = [7826.07, 7811.77, 7800.37, 7799.80, 7799.77]

# Data for FILS
durations_fils = [0.011, 0.012, 0.015, 0.029, 0.039]
objective_fils = [18205.00, 16814.00, 14171.00, 13177.00, 11439.00]

# Creating a plot with duration on x-axis and objective value on y-axis
plt.figure(figsize=(10,6))

# Plot for BILS
plt.plot(durations_bils, objective_bils, marker='o', label='BILS', linestyle='-', color='blue')

# Plot for RFILS
plt.plot(durations_rfils, objective_rfils, marker='o', label='RFILS', linestyle='-', color='green')

# Plot for FILS
plt.plot(durations_fils, objective_fils, marker='o', label='FILS', linestyle='-', color='red')

# Adding labels and title
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Duration for BILS, FILS, and RFILS')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()