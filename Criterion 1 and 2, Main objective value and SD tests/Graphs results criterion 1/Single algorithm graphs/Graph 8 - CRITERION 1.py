import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 8 provided results
time_points_rfi = [0.056, 0.093, 0.146, 0.180, 0.236]
mean_objective_rfi_new = [383.53, 377.03, 372.10, 370.87, 370.10]

# Data for Omega-W (as provided for Dataset 8)
time_points_omega_w_new = [0.066, 0.088, 0.146, 0.170, 0.233]
mean_objective_omega_w_dataset8 = [383.83, 383.83, 349.17, 349.17, 336.30]

# Data for Tabu Search (Dataset 8)
time_points_tabu_search = [0.062, 0.088, 0.124, 0.164, 0.237]
mean_objective_tabu_search = [371.40, 358.20, 347.03, 337.43, 322.77]

# Corrected SA data for Dataset 8 based on the new information
time_points_sim_annealing = [0.064, 0.100, 0.145, 0.190, 0.236]
mean_objective_sim_annealing = [364.60, 335.23, 323.83, 318.53, 314.97]

# GLS-Q data for Dataset 8
time_points_gls_1 = [0.055, 0.100, 0.135, 0.186, 0.236]
mean_objective_gls_1 = [391.47, 384.33, 372.33, 346.07, 336.37]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 8)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset8, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 8)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 8)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 8)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=252, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 8)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()