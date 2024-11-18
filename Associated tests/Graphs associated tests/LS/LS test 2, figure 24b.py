import matplotlib.pyplot as plt

# Data for RFILS with equal move acceptance
durations_rfils = [0.059, 0.127, 0.160, 0.206, 0.228, 1.010]  # Added one more data point for RFILS
objective_rfils = [229.27, 226.10, 225.53, 225.07, 224.90, 195.77]

# Data for RFILS without equal move acceptance (Dataset 7 results)
durations_rfils_no_equal_move = [0.058, 0.092, 0.175, 0.235, 1.022]  # Dataset 7 durations
objective_rfils_no_equal_move = [247.83, 244.27, 239.97, 239.13, 238.20]  # Dataset 7 mean objective values

# Creating a plot with duration on the x-axis and objective value on the y-axis
plt.figure(figsize=(10,6))
# Plot for RFILS
plt.plot(durations_rfils, objective_rfils, marker='o', label='RFILS (with immediate acceptance equal move)', linestyle='-', color='green')

# Plot for RFILS without equal move acceptance
plt.plot(durations_rfils_no_equal_move, objective_rfils_no_equal_move, marker='o', label='RFILS (no equal move accepted until 10 non-improving iterations limit reached)', linestyle='-', color='purple')

# Adding labels and title
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Duration for RFILS with and without accepting equal moves immediately (Dataset 7)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()