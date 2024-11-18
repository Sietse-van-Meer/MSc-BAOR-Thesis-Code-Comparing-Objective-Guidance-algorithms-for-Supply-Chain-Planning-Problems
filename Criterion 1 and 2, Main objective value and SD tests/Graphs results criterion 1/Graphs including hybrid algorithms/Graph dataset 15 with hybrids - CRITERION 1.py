import matplotlib.pyplot as plt

# Data for RFILS (Dataset 15)
time_points_rfi = [0.549, 1.229, 1.568, 1.875, 2.201]
mean_objective_rfi_new = [2015.77, 1998.80, 1995.80, 1994.00, 1993.90]

# Data for Omega-W (Dataset 15)
time_points_omega_w_new = [0.650, 0.779, 1.386, 1.535, 2.161]
mean_objective_omega_w_dataset15 = [2009.57, 2009.57, 1867.60, 1867.60, 1822.10]

# Data for Simulated Annealing (Dataset 15)
time_points_sim_annealing = [0.621, 1.068, 1.467, 1.818, 2.197]
mean_objective_sim_annealing = [1739.40, 1657.83, 1630.13, 1611.63, 1596.70]

# Data for LS + SA (Dataset 15)
time_points_ls_sa = [0.585, 0.757, 1.145, 1.525, 2.204]
mean_objective_ls_sa = [2014.97, 2008.60, 1758.33, 1655.50, 1599.43]

# Data for Omega-w + SA (Dataset 15)
time_points_omega_w_sa = [0.645, 0.783, 1.392, 1.814, 2.260]
mean_objective_omega_w_sa = [2009.57, 2009.57, 1867.60, 1835.70, 1657.77]

# Plotting the graph for Dataset 15
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 15)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 15)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset15, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 15)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 15)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 15)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 15)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
