import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 9 provided results
time_points_rfi = [0.067, 0.101, 0.163, 0.195, 0.255]
mean_objective_rfi_new = [547457.10, 547454.97, 547444.27, 547443.10, 547436.73]

# Data for Omega-W (as provided for Dataset 9)
time_points_omega_w_new = [0.068, 0.099, 0.159, 0.183, 0.244]
mean_objective_omega_w_dataset9 = [547456.93, 545289.53, 543356.27, 543284.20, 542900.83]

# Data for Simulated Annealing (Dataset 9)
time_points_sim_annealing = [0.067, 0.106, 0.150, 0.194, 0.249]
mean_objective_sim_annealing = [541448.47, 541407.30, 541394.93, 541389.93, 541385.30]

# Data for LS + SA (Dataset 9)
time_points_ls_sa = [0.066, 0.097, 0.127, 0.178, 0.254]
mean_objective_ls_sa = [546482.37, 542723.03, 541450.60, 541410.27, 541398.60]

# Updated Data for Omega-w + SA (Dataset 9)
time_points_omega_w_sa = [0.073, 0.100, 0.165, 0.194, 0.245]
mean_objective_omega_w_sa = [547456.93, 545289.53, 543356.27, 542165.43, 541378.50]

# Plotting the graph for Dataset 9
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 9)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset9, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 9)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 9)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 9)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=541208, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 9)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()