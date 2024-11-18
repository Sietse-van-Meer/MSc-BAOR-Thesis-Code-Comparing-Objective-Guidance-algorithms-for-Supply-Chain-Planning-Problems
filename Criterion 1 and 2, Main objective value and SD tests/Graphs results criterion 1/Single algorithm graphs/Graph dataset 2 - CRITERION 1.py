import matplotlib.pyplot as plt

# Data for RFILS (Dataset 2)
time_points_rfi = [0.009, 0.016, 0.025, 0.031, 0.038]
mean_objective_rfi_new = [269.87, 267.23, 266.53, 266.47, 266.40]

# Data for Omega-W (Dataset 2)
time_points_omega_w_new = [0.009, 0.014, 0.024, 0.027, 0.036]
mean_objective_omega_w_dataset2 = [270.20, 270.20, 232.23, 232.23, 224.07]

# Data for Tabu Search (Dataset 2)
time_points_tabu_search = [0.007, 0.013, 0.022, 0.030, 0.037]
mean_objective_tabu_search = [276.60, 260.23, 250.57, 239.87, 232.60]

# Data for Simulated Annealing (Dataset 2)
time_points_sim_annealing = [0.008, 0.013, 0.021, 0.028, 0.036]
mean_objective_sim_annealing = [775.17, 600.30, 243.30, 216.67, 208.63]

# Data for GLS (Dataset 2)
time_points_gls_2 = [0.008, 0.016, 0.024, 0.032, 0.039]
mean_objective_gls_2 = [272.80, 270.20, 251.00, 227.47, 224.57]

# Plotting the graph for Dataset 2
plt.figure(figsize=(10, 6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 2)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset2, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 2)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 2)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS (Dataset 2)
plt.plot(time_points_gls_2, mean_objective_gls_2, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=132, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 2)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()