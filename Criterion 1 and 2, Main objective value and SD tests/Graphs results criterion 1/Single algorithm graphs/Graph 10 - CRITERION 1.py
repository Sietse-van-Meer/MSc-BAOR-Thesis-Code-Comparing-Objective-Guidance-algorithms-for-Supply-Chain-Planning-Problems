import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 10 provided results
time_points_rfi = [0.100, 0.146, 0.202, 0.277, 0.344]
mean_objective_rfi_new = [482.77, 475.20, 468.80, 465.10, 462.10]

# Data for Omega-W (as provided for Dataset 10)
time_points_omega_w_new = [0.109, 0.133, 0.222, 0.243, 0.335]
mean_objective_omega_w_dataset10 = [485.97, 485.97, 478.50, 478.50, 475.97]

# Data for Tabu Search (Dataset 10)
time_points_tabu_search = [0.108, 0.147, 0.219, 0.284, 0.342]
mean_objective_tabu_search = [491.10, 486.83, 480.97, 476.77, 472.63]

# Corrected SA data for Dataset 10 based on the new information
time_points_sim_annealing = [0.107, 0.144, 0.193, 0.248, 0.343]
mean_objective_sim_annealing = [508.83, 497.87, 492.07, 488.10, 483.93]

# GLS-Q data for Dataset 10
time_points_gls_1 = [0.106, 0.173, 0.225, 0.279, 0.338]
mean_objective_gls_1 = [495.83, 486.63, 486.17, 485.97, 485.97]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 10)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset10, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 10)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 10)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 10)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=407, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 10)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()