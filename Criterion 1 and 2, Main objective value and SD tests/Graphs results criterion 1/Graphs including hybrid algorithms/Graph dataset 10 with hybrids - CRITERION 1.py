import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 10 provided results
time_points_rfi = [0.100, 0.146, 0.202, 0.277, 0.344]
mean_objective_rfi_new = [482.77, 475.20, 468.80, 465.10, 462.10]

# Data for Omega-W (as provided for Dataset 10)
time_points_omega_w_new = [0.109, 0.133, 0.222, 0.243, 0.335]
mean_objective_omega_w_dataset10 = [485.97, 485.97, 478.50, 478.50, 475.97]

# Data for Simulated Annealing (Dataset 10)
time_points_sim_annealing = [0.107, 0.144, 0.193, 0.248, 0.343]
mean_objective_sim_annealing = [508.83, 497.87, 492.07, 488.10, 483.93]

# Data for LS + SA (Dataset 10)
time_points_ls_sa = [0.106, 0.145, 0.195, 0.240, 0.338]
mean_objective_ls_sa = [486.90, 486.03, 484.17, 480.87, 475.77]

# Data for Omega-w + SA (Dataset 10)
time_points_omega_w_sa = [0.108, 0.129, 0.240, 0.297, 0.345]
mean_objective_omega_w_sa = [485.97, 485.97, 478.50, 478.50, 478.47]

# Plotting the graph for Dataset 10
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 10)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset10, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 10)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 10)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 10)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=407, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 10)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
