import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 6 provided results
time_points_rfi = [0.007, 0.013, 0.021, 0.025, 0.033]
mean_objective_rfi_new = [32199.57, 32186.17, 32184.93, 32184.73, 32184.73]

# Data for Omega-W (as provided for Dataset 6)
time_points_omega_w_new = [0.007, 0.015, 0.021, 0.026, 0.032]
mean_objective_omega_w_dataset6 = [32204.87, 25816.50, 25457.83, 25453.50, 25414.37]

# Data for Tabu Search (Dataset 6)
time_points_tabu_search = [0.006, 0.011, 0.018, 0.023, 0.032]
mean_objective_tabu_search = [31818.70, 30796.73, 30231.07, 29718.00, 29178.30]

# Corrected SA data for Dataset 6 based on the new information
time_points_sim_annealing = [0.007, 0.014, 0.020, 0.027, 0.035]
mean_objective_sim_annealing = [28683.00, 25055.20, 24827.83, 24817.07, 24811.53]

# GLS-Q data for Dataset 6
time_points_gls_1 = [0.006, 0.011, 0.018, 0.026, 0.034]
mean_objective_gls_1 = [32232.43, 31708.30, 28453.50, 26410.50, 25685.00]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 6)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset6, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 6)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 6)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 6)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=24372, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 6)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()