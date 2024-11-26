import matplotlib.pyplot as plt

# Data for BILS
durations_bils = [0.054, 0.101, 0.141, 0.185, 0.228, 1.069]  # Added one more data point for BILS
objective_bils = [5422.00, 2492.50, 2088.17, 1696.87, 1399.07, 570.70]

# Data for RFILS
durations_rfils = [0.059, 0.127, 0.160, 0.206, 0.228, 1.010]  # Added one more data point for RFILS
objective_rfils = [229.27, 226.10, 225.53, 225.07, 224.90, 195.77]

# Data for FILS
durations_fils = [0.057, 0.083, 0.128, 0.183, 0.225, 1.057]  # Added one more data point for FILS
objective_fils = [10488.00, 3266.00, 2482.00, 1475.00, 1223.00, 619.77]

# Creating a plot with duration on the x-axis and objective value on the y-axis
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
plt.title('Mean Objective Value vs Duration for BILS, FILS, and RFILS (Dataset 7)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()