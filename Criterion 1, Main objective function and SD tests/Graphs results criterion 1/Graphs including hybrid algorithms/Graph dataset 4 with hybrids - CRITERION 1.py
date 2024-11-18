import matplotlib.pyplot as plt

# Data for RFILS (Dataset 4)
time_points_rfi = [0.007, 0.013, 0.021, 0.027, 0.031]
mean_objective_rfi_new = [33250.47, 33249.27, 33249.10, 33249.10, 33249.10]

# Data for Omega-W (Dataset 4)
time_points_omega_w_new = [0.007, 0.015, 0.021, 0.026, 0.030]
mean_objective_omega_w_dataset4 = [33259.07, 31446.00, 30974.23, 30974.23, 30917.33]

# Data for Simulated Annealing (Dataset 4)
time_points_sim_annealing = [0.006, 0.014, 0.020, 0.025, 0.030]
mean_objective_sim_annealing = [33262.00, 30851.63, 30653.63, 30644.40, 30637.57]

# Data for LS + SA (Dataset 4)
time_points_ls_sa = [0.006, 0.010, 0.018, 0.022, 0.030]
mean_objective_ls_sa = [33263.37, 33089.07, 31048.77, 30737.07, 30656.20]

# Updated Data for Omega-w + SA (Dataset 4)
time_points_omega_w_sa = [0.007, 0.014, 0.020, 0.026, 0.032]
mean_objective_omega_w_sa = [33259.07, 31446.00, 30974.23, 30974.23, 30886.27]

# Plotting the graph for Dataset 4
plt.figure(figsize=(10, 6))

# Plot for RFILS (Dataset 4)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 4)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset4, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 4)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 4)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 4)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=30549, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 4)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
