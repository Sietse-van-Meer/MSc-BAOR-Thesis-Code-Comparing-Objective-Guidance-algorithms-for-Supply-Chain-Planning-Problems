import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 7 provided results
time_points_rfi = [0.054, 0.090, 0.148, 0.184, 0.238]
mean_objective_rfi_new = [229.30, 226.87, 225.47, 225.07, 224.77]

# Updated data for Omega-W (Dataset 7)
time_points_omega_w_7 = [0.060, 0.083, 0.146, 0.161, 0.225]
mean_objective_omega_w_7 = [229.23, 229.23, 196.40, 196.40, 182.13]

# Corrected SA data for Dataset 7 based on the new information
time_points_sim_annealing = [0.054, 0.085, 0.143, 0.186, 0.225]
mean_objective_sim_annealing = [304.73, 213.47, 196.60, 190.03, 186.03]

# Data for LS + SA
time_points_ls_sa_7 = [0.060, 0.094, 0.125, 0.168, 0.235]
mean_objective_ls_sa_7 = [229.60, 224.20, 183.87, 165.53, 158.20]

# Data for Omega-w + SA
time_points_omega_w_sa = [0.058, 0.082, 0.145, 0.194, 0.234]
mean_objective_omega_w_sa = [229.23, 229.23, 196.40, 196.40, 165.57]
# Plotting the graph for Dataset 7
plt.figure(figsize=(10,6))

# Plot for RFI (Dataset 7)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 7)
plt.plot(time_points_omega_w_7, mean_objective_omega_w_7, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 7)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 7)
plt.plot(time_points_ls_sa_7, mean_objective_ls_sa_7, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 7)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')


# Adding vertical line for ILP solution
plt.axhline(y=75, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for RFI, Omega-W, Tabu Search, and Simulated Annealing (Dataset 7)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()