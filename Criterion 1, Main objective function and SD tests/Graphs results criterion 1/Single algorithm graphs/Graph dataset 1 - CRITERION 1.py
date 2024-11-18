import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on your provided results
time_points_rfi = [0.007, 0.017, 0.024, 0.030, 0.037]
mean_objective_rfi_new = [7828.43, 7800.57, 7800.37, 7800.33, 7800.20]

# Data for Omega-W (as you provided for Dataset 1)
time_points_omega_w_new = [0.009, 0.017, 0.025, 0.031, 0.037]
mean_objective_omega_w_dataset1 = [7826.07, 6262.77, 5943.03, 5796.13, 5699.50]

# Corrected data for Tabu Search (Dataset 1)
time_points_tabu_search = [0.008, 0.012, 0.017, 0.025, 0.039]
mean_objective_tabu_search = [7636.17, 7352.40, 6981.37, 6726.33, 6472.23]

# Corrected data for Simulated Annealing (Dataset 1)
time_points_sim_annealing = [0.007, 0.015, 0.020, 0.029, 0.039]
mean_objective_sim_annealing = [8155.87, 5055.13, 4944.23, 4930.87, 4866.53]

# Updated data for GLS (Dataset 1)
time_points_gls_1 = [0.009, 0.015, 0.022, 0.029, 0.039]
mean_objective_gls_1 = [7846.20, 7160.33, 6223.00, 5843.60, 5499.80]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 1)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset1, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (updated)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (updated)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS (Dataset 1)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=4372, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 1)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()