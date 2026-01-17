import matplotlib.pyplot as plt

# Data for RFILS (Dataset 1)
time_points_rfi = [0.007, 0.017, 0.024, 0.030, 0.037]
mean_objective_rfi_new = [7828.43, 7800.57, 7800.37, 7800.33, 7800.20]

# Data for Omega-W (Dataset 1)
time_points_omega_w_new = [0.009, 0.017, 0.025, 0.031, 0.037]
mean_objective_omega_w_dataset1 = [7826.07, 6262.77, 5943.03, 5796.13, 5699.50]

# Data for Simulated Annealing (Dataset 1)
time_points_sim_annealing = [0.007, 0.015, 0.020, 0.029, 0.039]
mean_objective_sim_annealing = [8155.87, 5055.13, 4944.23, 4930.87, 4866.53]

# Data for LS + SA (Dataset 1)
time_points_ls_sa = [0.008, 0.013, 0.022, 0.029, 0.039]
mean_objective_ls_sa = [7829.03, 7424.80, 5298.57, 5029.63, 4963.97]

# Data for Omega-w + SA (Dataset 1)
time_points_omega_w_sa = [0.008, 0.017, 0.026, 0.034, 0.037]
mean_objective_omega_w_sa = [7826.07, 6262.77, 5943.03, 5943.03, 5244.90]

# Plotting the graph for Dataset 1
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 1)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 1)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset1, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 1)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 1)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 1)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=4372, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 1)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
