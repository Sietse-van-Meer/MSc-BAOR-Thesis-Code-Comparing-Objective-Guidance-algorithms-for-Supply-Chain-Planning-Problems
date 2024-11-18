import matplotlib.pyplot as plt

# Data for RFILS (Dataset 5)
time_points_rfi = [0.006, 0.014, 0.024, 0.030, 0.037]
mean_objective_rfi_new = [46447.10, 46437.03, 46436.57, 46436.57, 46436.43]

# Data for Omega-W (Dataset 5)
time_points_omega_w_new = [0.006, 0.014, 0.022, 0.026, 0.034]
mean_objective_omega_w_dataset5 = [46465.07, 43866.83, 43191.43, 43191.43, 43156.93]

# Data for Tabu Search (Dataset 5)
time_points_tabu_search = [0.006, 0.014, 0.019, 0.025, 0.034]
mean_objective_tabu_search = [45996.00, 45481.73, 45237.60, 44930.47, 44644.30]

# Data for Simulated Annealing (Dataset 5)
time_points_sim_annealing = [0.005, 0.014, 0.020, 0.026, 0.034]
mean_objective_sim_annealing = [46351.77, 43631.40, 43417.13, 43406.43, 43401.50]

# Data for GLS-Q (Dataset 5)
time_points_gls_5 = [0.006, 0.011, 0.017, 0.024, 0.034]
mean_objective_gls_5 = [46472.13, 46405.33, 44735.27, 43968.90, 43598.47]

# Plotting the graph for Dataset 5
plt.figure(figsize=(10, 6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 5)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset5, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 5)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 5)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 5)
plt.plot(time_points_gls_5, mean_objective_gls_5, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=42767, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 5)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()