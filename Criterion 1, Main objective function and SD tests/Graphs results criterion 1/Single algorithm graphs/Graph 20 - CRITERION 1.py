import matplotlib.pyplot as plt

# Data for RFILS (Dataset 20)
time_points_rfi = [1.475, 2.941, 4.403, 5.575, 6.876]
mean_objective_rfi = [485046.40, 485000.03, 484964.73, 484961.07, 484960.43]

# Data for Omega-W (Dataset 20)
time_points_omega_w = [1.525, 2.751, 4.441, 5.004, 6.519]
mean_objective_omega_w = [485044.33, 95023.13, 82245.57, 82245.57, 81830.70]

# Data for Tabu Search (Dataset 20)
time_points_tabu_search = [1.331, 2.385, 3.812, 5.303, 6.528]
mean_objective_tabu_search = [438423.97, 399110.67, 380732.77, 367021.33, 333393.23]

# Data for Simulated Annealing (Dataset 20)
time_points_sim_annealing = [1.424, 2.364, 3.554, 4.949, 6.597]
mean_objective_sim_annealing = [73989.57, 73530.63, 73344.77, 73235.23, 73146.07]

# Data for GLS-Q (Dataset 20)
time_points_gls_q = [1.444, 2.449, 4.068, 5.560, 6.810]
mean_objective_gls_q = [485136.27, 461843.83, 137810.90, 90059.33, 75743.27]

# Plotting the graph for Dataset 20
plt.figure(figsize=(12, 8))

# Plot for RFILS (Dataset 20)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 20)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 20)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 20)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 20)
plt.plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 20)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()