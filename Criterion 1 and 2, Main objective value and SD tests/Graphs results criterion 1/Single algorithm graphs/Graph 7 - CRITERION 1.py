import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 7 provided results
time_points_rfi = [0.054, 0.090, 0.148, 0.184, 0.238]
mean_objective_rfi_new = [229.30, 226.87, 225.47, 225.07, 224.77]

# Data for Omega-W (as provided for Dataset 7)
time_points_omega_w_new = [0.060, 0.083, 0.146, 0.161, 0.225]
mean_objective_omega_w_dataset7 = [229.23, 229.23, 196.40, 196.40, 182.13]

# Data for Tabu Search (Dataset 7)
time_points_tabu_search = [0.060, 0.089, 0.123, 0.163, 0.226]
mean_objective_tabu_search = [243.50, 228.20, 217.10, 199.63, 182.83]

# Corrected SA data for Dataset 7 based on the new information
time_points_sim_annealing = [0.054, 0.085, 0.143, 0.186, 0.225]
mean_objective_sim_annealing = [304.73, 213.47, 196.60, 190.03, 186.03]

# GLS-Q data for Dataset 7
time_points_gls_1 = [0.055, 0.105, 0.138, 0.188, 0.230]
mean_objective_gls_1 = [233.60, 229.33, 228.33, 199.73, 185.73]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 7)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset7, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 7)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 7)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 7)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=75, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 7)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()