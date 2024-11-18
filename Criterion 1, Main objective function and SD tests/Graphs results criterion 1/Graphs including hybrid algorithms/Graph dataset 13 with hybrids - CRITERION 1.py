import matplotlib.pyplot as plt

# Data for RFILS (Dataset 13)
time_points_rfi = [0.423, 0.495, 0.906, 1.180, 1.470]
mean_objective_rfi_new = [2405.60, 2401.17, 2398.60, 2397.70, 2397.50]

# Data for Omega-W (Dataset 13)
time_points_omega_w_new = [0.399, 0.526, 0.889, 1.032, 1.379]
mean_objective_omega_w_dataset13 = [2403.53, 2403.53, 2362.63, 2362.63, 2357.87]

# Data for Simulated Annealing (Dataset 13)
time_points_sim_annealing = [0.371, 0.565, 0.795, 1.058, 1.423]
mean_objective_sim_annealing = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]

# Data for LS + SA (Dataset 13)
time_points_ls_sa = [0.372, 0.566, 0.843, 1.038, 1.396]
mean_objective_ls_sa = [2404.93, 2403.53, 2163.20, 2069.67, 2040.07]

# Data for Omega-w + SA (Dataset 13)
time_points_omega_w_sa = [0.539, 0.562, 0.946, 1.133, 1.410]
mean_objective_omega_w_sa = [2403.53, 2403.53, 2362.63, 2362.63, 2135.07]

# Plotting the graph for Dataset 13
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 13)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 13)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset13, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 13)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 13)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 13)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 13)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()