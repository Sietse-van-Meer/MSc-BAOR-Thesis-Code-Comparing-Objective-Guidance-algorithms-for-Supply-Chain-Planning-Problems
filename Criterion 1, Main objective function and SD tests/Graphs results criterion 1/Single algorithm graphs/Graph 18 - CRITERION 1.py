import matplotlib.pyplot as plt

# Data for RFILS (Dataset 18)
time_points_rfi = [0.439, 0.749, 1.121, 1.450, 1.704]
mean_objective_rfi = [5293.63, 5277.97, 5273.70, 5271.30, 5270.27]

# Data for Omega-W (Dataset 18)
time_points_omega_w = [0.544, 0.646, 1.116, 1.205, 1.686]
mean_objective_omega_w = [5288.93, 5288.93, 5264.37, 5264.37, 5263.50]

# Data for Tabu Search (Dataset 18)
time_points_tabu_search = [0.516, 0.966, 1.105, 1.378, 1.769]
mean_objective_tabu_search = [5312.90, 5298.23, 5296.53, 5291.77, 5290.67]

# Data for Simulated Annealing (Dataset 18)
time_points_sim_annealing = [0.517, 0.845, 1.073, 1.253, 1.716]
mean_objective_sim_annealing = [4592.87, 4466.63, 4400.33, 4371.40, 4315.47]

# Data for GLS-Q (Dataset 18)
time_points_gls_q = [0.534, 0.749, 1.142, 1.393, 1.783]
mean_objective_gls_q = [5304.07, 5292.57, 5184.97, 5122.37, 5065.93]

# Plotting the graph for Dataset 18
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 18)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 18)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 18)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 18)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 18)
plt.plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 18)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()