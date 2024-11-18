import matplotlib.pyplot as plt

# Data for RFILS (Dataset 17)
time_points_rfi = [0.714, 1.189, 1.461, 2.085, 2.510]
mean_objective_rfi = [2537.77, 2522.50, 2518.53, 2514.03, 2513.33]

# Data for Omega-W (Dataset 17)
time_points_omega_w = [0.881, 0.976, 1.577, 1.704, 2.308]
mean_objective_omega_w = [2535.17, 2535.17, 2513.90, 2513.90, 2513.73]

# Data for Tabu Search (Dataset 17)
time_points_tabu_search = [0.810, 1.035, 1.328, 1.800, 2.329]
mean_objective_tabu_search = [2521.97, 2518.70, 2512.40, 2506.43, 2500.83]

# Data for Simulated Annealing (Dataset 17)
time_points_sim_annealing = [0.808, 1.138, 1.395, 1.772, 2.372]
mean_objective_sim_annealing = [2347.53, 2300.07, 2269.87, 2236.70, 2196.27]

# Data for GLS-Q (Dataset 17)
time_points_gls_q = [0.855, 1.134, 1.434, 1.856, 2.347]
mean_objective_gls_q = [2550.77, 2529.53, 2523.60, 2512.40, 2510.17]

# Plotting the graph for Dataset 17
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 17)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 17)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 17)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 17)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 17)
plt.plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 17)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()