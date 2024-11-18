import matplotlib.pyplot as plt

# Updated Data for RFILS (Dataset 20)
time_points_rfi = [1.475, 2.941, 4.403, 5.575, 6.876]
mean_objective_rfi = [485046.40, 485000.03, 484964.73, 484961.07, 484960.43]

# Updated Data for Omega-W (Dataset 20)
time_points_omega_w = [1.525, 2.751, 4.441, 5.004, 6.519]
mean_objective_omega_w = [485044.33, 95023.13, 82245.57, 82245.57, 81830.70]

# Updated Data for Simulated Annealing (SA) (Dataset 20)
time_points_sim_annealing = [1.424, 2.364, 3.554, 4.949, 6.597]
mean_objective_sim_annealing = [73989.57, 73530.63, 73344.77, 73235.23, 73146.07]

# Updated Data for LS + SA (Dataset 20)
time_points_ls_sa = [1.474, 2.737, 4.059, 5.476, 6.584]
mean_objective_ls_sa = [367770.90, 74676.97, 73422.70, 73231.43, 73148.00]

# Updated Data for Omega-w + SA (Dataset 20)
time_points_omega_sa = [1.667, 3.011, 4.708, 5.509, 6.846]
mean_objective_omega_sa = [485044.33, 95023.13, 82245.57, 74871.57, 73412.57]

# Plotting the graph for Dataset 20
plt.figure(figsize=(10, 6))

# Plot for RFILS (Dataset 20)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 20)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 20)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 20)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 20)
plt.plot(time_points_omega_sa, mean_objective_omega_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 20)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
