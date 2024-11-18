import matplotlib.pyplot as plt

# Data for RFILS with equal move acceptance (Dataset 1)
durations_rfils = [0.007, 0.011, 0.025, 0.035, 0.043]
objective_rfils = [7826.07, 7811.77, 7800.37, 7799.80, 7799.77]

# Data for RFILS without equal move acceptance (Dataset 1 results)
durations_rfils_no_equal_move = [0.008, 0.018, 0.028, 0.035, 0.038]  # Dataset 1 durations
objective_rfils_no_equal_move = [7918.73, 7887.83, 7886.97, 7886.67, 7886.67]  # Dataset 1 mean objective values

# Creating a plot with duration on the x-axis and objective value on the y-axis
plt.figure(figsize=(10,6))

# Plot for RFILS with equal move acceptance
plt.plot(durations_rfils, objective_rfils, marker='o', label='RFILS (with immediate acceptance equal move)', linestyle='-', color='green')

# Plot for RFILS without equal move acceptance
plt.plot(durations_rfils_no_equal_move, objective_rfils_no_equal_move, marker='o', label='RFILS (no equal move accepted until 10 non-improving iterations limit reached)', linestyle='-', color='purple')

# Adding labels and title
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Duration for RFILS with and without accepting equal moves immediately (Dataset 1)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()