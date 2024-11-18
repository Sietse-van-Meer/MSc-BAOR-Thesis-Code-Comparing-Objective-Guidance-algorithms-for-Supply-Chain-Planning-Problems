import matplotlib.pyplot as plt

# Data for RFILS (Dataset 13)
time_points_rfi = [0.423, 0.495, 0.906, 1.180, 1.470]
mean_objective_rfi_new = [2405.60, 2401.17, 2398.60, 2397.70, 2397.50]

# Data for Omega-W (Dataset 13)
time_points_omega_w_new = [0.399, 0.526, 0.889, 1.032, 1.379]
mean_objective_omega_w_dataset13 = [2403.53, 2403.53, 2362.63, 2362.63, 2357.87]

# Data for Tabu Search (Dataset 13)
time_points_tabu_search = [0.386, 0.585, 0.822, 1.140, 1.388]
mean_objective_tabu_search = [2394.93, 2380.77, 2372.50, 2364.90, 2349.87]

# Corrected SA data for Dataset 13
time_points_sim_annealing = [0.371, 0.565, 0.795, 1.058, 1.423]
mean_objective_sim_annealing = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]

# GLS-Q data for Dataset 13
time_points_gls_1 = [0.399, 0.593, 0.781, 1.011, 1.465]
mean_objective_gls_1 = [2408.27, 2404.50, 2395.97, 2379.47, 2348.17]

# Plotting the graph for Dataset 13
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 13)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 13)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset13, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 13)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 13)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 13)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 13)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()