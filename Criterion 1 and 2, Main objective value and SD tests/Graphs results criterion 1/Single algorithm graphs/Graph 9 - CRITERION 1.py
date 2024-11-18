import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 9 provided results
time_points_rfi = [0.067, 0.101, 0.163, 0.195, 0.255]
mean_objective_rfi_new = [547457.10, 547454.97, 547444.27, 547443.10, 547436.73]

# Data for Omega-W (as provided for Dataset 9)
time_points_omega_w_new = [0.068, 0.099, 0.159, 0.183, 0.244]
mean_objective_omega_w_dataset9 = [547456.93, 545289.53, 543356.27, 543284.20, 542900.83]

# Data for Tabu Search (Dataset 9)
time_points_tabu_search = [0.066, 0.113, 0.145, 0.207, 0.248]
mean_objective_tabu_search = [546376.03, 545660.43, 545424.43, 544728.63, 544622.47]

# Corrected SA data for Dataset 9 based on the new information
time_points_sim_annealing = [0.067, 0.106, 0.150, 0.194, 0.249]
mean_objective_sim_annealing = [541448.47, 541407.30, 541394.93, 541389.93, 541385.30]

# GLS-Q data for Dataset 9
time_points_gls_1 = [0.064, 0.105, 0.140, 0.196, 0.250]
mean_objective_gls_1 = [547480.90, 547295.87, 546513.27, 544966.00, 543842.70]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 9)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset9, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 9)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 9)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 9)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=541208, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 9)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()