import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 12 provided results
time_points_rfi = [0.048, 0.078, 0.109, 0.143, 0.188]
mean_objective_rfi_new = [277.63, 275.57, 275.00, 274.30, 274.13]

# Data for Omega-W (as provided for Dataset 12)
time_points_omega_w_new = [0.051, 0.070, 0.116, 0.135, 0.184]
mean_objective_omega_w_dataset12 = [278.07, 278.07, 257.77, 257.77, 250.87]

# Data for Tabu Search (Dataset 12)
time_points_tabu_search = [0.047, 0.076, 0.108, 0.147, 0.190]
mean_objective_tabu_search = [277.73, 270.83, 262.13, 254.23, 248.47]

# Corrected SA data for Dataset 12 based on the new information
time_points_sim_annealing = [0.048, 0.071, 0.098, 0.152, 0.186]
mean_objective_sim_annealing = [487.27, 277.47, 264.03, 257.07, 254.40]

# GLS-Q data for Dataset 12
time_points_gls_1 = [0.049, 0.091, 0.121, 0.156, 0.184]
mean_objective_gls_1 = [280.40, 278.10, 265.03, 252.40, 247.93]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 12)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset12, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 12)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 12)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 12)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=184, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 12)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()