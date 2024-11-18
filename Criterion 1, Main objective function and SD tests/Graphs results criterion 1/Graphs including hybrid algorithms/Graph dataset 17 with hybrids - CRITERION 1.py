import matplotlib.pyplot as plt

# Data for RFILS (Dataset 17)
time_points_rfi = [0.714, 1.189, 1.461, 2.085, 2.510]
mean_objective_rfi = [2537.77, 2522.50, 2518.53, 2514.03, 2513.33]

# Data for Omega-W (Dataset 17)
time_points_omega_w = [0.881, 0.976, 1.577, 1.704, 2.308]
mean_objective_omega_w = [2535.17, 2535.17, 2513.90, 2513.90, 2513.73]

# Data for Simulated Annealing (Dataset 17)
time_points_sim_annealing = [0.808, 1.138, 1.395, 1.772, 2.372]
mean_objective_sim_annealing = [2347.53, 2300.07, 2269.87, 2236.70, 2196.27]

# Data for LS + SA (Dataset 17)
time_points_ls_sa = [0.800, 1.183, 1.525, 1.768, 2.329]
mean_objective_ls_sa = [2523.17, 2392.73, 2269.17, 2228.67, 2177.63]

# Data for Omega-w + SA (Dataset 17)
time_points_omega_w_sa = [0.982, 0.947, 1.544, 1.965, 2.349]
mean_objective_omega_w_sa = [2535.17, 2535.17, 2513.90, 2500.13, 2287.80]

# Plotting the graph for Dataset 17
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 17)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 17)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 17)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 17)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 17)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 17)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
