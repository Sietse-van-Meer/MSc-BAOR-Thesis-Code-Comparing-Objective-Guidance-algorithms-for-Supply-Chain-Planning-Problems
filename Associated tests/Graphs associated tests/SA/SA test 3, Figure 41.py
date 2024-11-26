import matplotlib.pyplot as plt

# Data for SA with cooling rate (Dataset 13)
durations_with_cooling_13 = [0.371, 0.565, 0.795, 1.058, 1.423]
objective_with_cooling_13 = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]

# Data for SA without cooling rate (Dataset 13)
durations_no_cooling_13 = [0.321, 0.504, 0.728, 0.996, 1.404]
objective_no_cooling_13 = [2251.60, 2227.47, 2203.27, 2183.80, 2168.20]

# Data for SA with cooling rate (Dataset 17)
durations_with_cooling_17 = [0.808, 1.138, 1.395, 1.772, 2.372]
objective_with_cooling_17 = [2347.53, 2300.07, 2269.87, 2236.70, 2196.27]

# Data for SA without cooling rate (Dataset 17)
durations_no_cooling_17 = [0.772, 1.095, 1.504, 1.718, 2.436]
objective_no_cooling_17 = [2457.47, 2397.83, 2362.60, 2327.53, 2281.87]

# Plotting the graphs
plt.figure(figsize=(14, 7))

# Plot for Dataset 13
plt.subplot(1, 2, 1)
plt.plot(durations_with_cooling_13, objective_with_cooling_13, marker='o', linestyle='-', color='blue', label='With Cooling Rate')
plt.plot(durations_no_cooling_13, objective_no_cooling_13, marker='o', linestyle='-', color='green', label='Without Cooling Rate')
plt.title('Dataset 13: SA With vs Without Cooling Rate')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()
plt.grid(True)

# Plot for Dataset 17
plt.subplot(1, 2, 2)
plt.plot(durations_with_cooling_17, objective_with_cooling_17, marker='o', linestyle='-', color='blue', label='With Cooling Rate')
plt.plot(durations_no_cooling_17, objective_no_cooling_17, marker='o', linestyle='-', color='green', label='Without Cooling Rate')
plt.title('Dataset 17: SA With vs Without Cooling Rate')
plt.xlabel('Average Duration per Run (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()
plt.grid(True)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()