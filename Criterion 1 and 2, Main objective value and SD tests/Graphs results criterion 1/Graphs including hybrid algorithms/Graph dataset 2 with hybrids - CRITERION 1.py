import matplotlib.pyplot as plt

# Data for RFILS (Dataset 2)
time_points_rfi = [0.009, 0.016, 0.025, 0.031, 0.038]
mean_objective_rfi_new = [269.87, 267.23, 266.53, 266.47, 266.40]

# Data for Omega-W (Dataset 2)
time_points_omega_w_new = [0.009, 0.014, 0.024, 0.027, 0.036]
mean_objective_omega_w_dataset2 = [270.20, 270.20, 232.23, 232.23, 224.07]

# Data for Simulated Annealing (Dataset 2)
time_points_sim_annealing = [0.008, 0.013, 0.021, 0.028, 0.036]
mean_objective_sim_annealing = [775.17, 600.30, 243.30, 216.67, 208.63]

# Data for LS + SA (Dataset 2)
time_points_ls_sa = [0.008, 0.014, 0.024, 0.027, 0.037]
mean_objective_ls_sa = [270.30, 270.20, 263.10, 226.23, 200.73]

# Updated Data for Omega-w + SA (Dataset 2)
time_points_omega_w_sa = [0.009, 0.014, 0.027, 0.032, 0.038]
mean_objective_omega_w_sa = [270.20, 270.20, 232.23, 232.23, 232.23]

# Plotting the graph for Dataset 2
plt.figure(figsize=(10, 6))

# Plot for RFILS (Dataset 2)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 2)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset2, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 2)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 2)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 2)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

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
