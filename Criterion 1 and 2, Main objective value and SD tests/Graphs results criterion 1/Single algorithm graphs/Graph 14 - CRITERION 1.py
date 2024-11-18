import matplotlib.pyplot as plt

# Data for RFILS (Dataset 14)
time_points_rfi = [1.105, 1.807, 2.594, 3.299, 4.043]
mean_objective_rfi_new = [5304.97, 5274.17, 5257.57, 5250.43, 5246.07]

# Data for Omega-W (Dataset 14)
time_points_omega_w_new = [1.184, 1.304, 2.432, 2.618, 3.821]
mean_objective_omega_w_dataset14 = [5310.33, 5310.33, 5089.23, 5089.23, 5037.87]

# Data for Tabu Search (Dataset 14)
time_points_tabu_search = [1.146, 1.851, 2.403, 2.946, 4.046]
mean_objective_tabu_search = [5347.87, 5339.40, 5304.30, 5280.97, 5267.17]

# Corrected SA data for Dataset 14
time_points_sim_annealing = [1.144, 1.858, 2.449, 3.170, 3.865]
mean_objective_sim_annealing = [4408.83, 4310.53, 4262.43, 4216.73, 4186.10]

# GLS-Q data for Dataset 14
time_points_gls_1 = [1.070, 1.511, 2.334, 3.067, 3.908]
mean_objective_gls_1 = [5356.03, 5302.53, 5103.47, 4965.60, 4881.47]

# Plotting the graph for Dataset 14
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 14)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 14)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset14, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 14)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 14)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 14)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 14)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()