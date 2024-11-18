import matplotlib.pyplot as plt

# Data for RFILS (Dataset 3)
time_points_rfi = [0.007, 0.014, 0.021, 0.029, 0.037]
mean_objective_rfi_new = [10441.03, 10406.33, 10406.33, 10406.33, 10403.33]

# Data for Omega-W (Dataset 3)
time_points_omega_w_new = [0.007, 0.017, 0.024, 0.029, 0.036]
mean_objective_omega_w_dataset3 = [10465.20, 9185.43, 8847.20, 8734.53, 8626.50]

# Data for Tabu Search (Dataset 3)
time_points_tabu_search = [0.007, 0.012, 0.020, 0.027, 0.036]
mean_objective_tabu_search = [10278.37, 10096.47, 9964.23, 9825.63, 9702.47]

# Data for Simulated Annealing (Dataset 3)
time_points_sim_annealing = [0.006, 0.011, 0.017, 0.028, 0.036]
mean_objective_sim_annealing = [13276.13, 8523.13, 7967.47, 7893.17, 7868.00]

# Data for GLS (Dataset 3)
time_points_gls_3 = [0.007, 0.015, 0.022, 0.030, 0.037]
mean_objective_gls_3 = [10519.30, 9787.97, 8881.60, 8589.70, 8301.90]

# Plotting the graph for Dataset 3
plt.figure(figsize=(10, 6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 3)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset3, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 3)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 3)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS (Dataset 3)
plt.plot(time_points_gls_3, mean_objective_gls_3, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=7505, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 3)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()