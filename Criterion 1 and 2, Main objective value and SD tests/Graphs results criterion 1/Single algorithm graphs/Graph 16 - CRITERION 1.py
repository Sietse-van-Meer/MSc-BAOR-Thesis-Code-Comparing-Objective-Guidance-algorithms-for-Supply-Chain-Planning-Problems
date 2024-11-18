import matplotlib.pyplot as plt

# Data for RFILS (Dataset 16)
time_points_rfi = [0.321, 0.612, 0.801, 1.056, 1.281]
mean_objective_rfi = [439328.00, 439270.40, 439268.57, 439265.97, 439264.83]

# Data for Omega-W (Dataset 16)
time_points_omega_w = [0.332, 0.600, 0.832, 0.936, 1.176]
mean_objective_omega_w = [439330.30, 283292.17, 281612.13, 281540.83, 280847.67]

# Data for Tabu Search (Dataset 16)
time_points_tabu_search = [0.330, 0.488, 0.640, 0.891, 1.219]
mean_objective_tabu_search = [428496.33, 405471.03, 390075.63, 374107.43, 357821.13]

# Data for Simulated Annealing (Dataset 16)
time_points_sim_annealing = [0.306, 0.484, 0.648, 0.868, 1.213]
mean_objective_sim_annealing = [248365.87, 248117.13, 248046.07, 247973.07, 247900.83]

# Data for GLS-Q (Dataset 16)
time_points_gls_q = [0.294, 0.431, 0.608, 0.908, 1.268]
mean_objective_gls_q = [439508.33, 439364.10, 407441.37, 294665.83, 269430.67]

# Plotting the graph for Dataset 16
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 16)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 16)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 16)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 16)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 16)
plt.plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 16)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()