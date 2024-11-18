import matplotlib.pyplot as plt

# Updated Data for RFILS (Dataset 19)
time_points_rfi = [3.413, 6.257, 8.228, 10.396, 13.130]
mean_objective_rfi = [7280657.27, 7280595.33, 7280585.40, 7280577.67, 7280575.80]

# Updated Data for Omega-W (Dataset 19)
time_points_omega_w = [3.509, 5.401, 8.318, 9.352, 12.610]
mean_objective_omega_w = [7280654.47, 4343605.63, 4320254.97, 4319965.13, 4316317.03]

# Updated Data for Simulated Annealing (SA) (Dataset 19)
time_points_sim_annealing = [3.486, 5.645, 8.143, 10.666, 13.168]
mean_objective_sim_annealing = [4264611.50, 4264456.50, 4264368.47, 4264317.43, 4264278.13]

# Updated Data for LS + SA (Dataset 19)
time_points_ls_sa = [3.268, 5.919, 8.393, 10.897, 12.899]
mean_objective_ls_sa = [5618376.97, 4268211.17, 4267271.40, 4267166.93, 4267109.50]

# Updated Data for Omega-w + SA (Dataset 19)
time_points_omega_sa = [3.885, 5.792, 8.991, 10.293, 12.738]
mean_objective_omega_sa = [7280654.47, 4343605.63, 4320254.97, 4185674.00, 4184725.37]

# Plotting the graph for Dataset 19
plt.figure(figsize=(12, 8))

# Plot for RFILS (Dataset 19)
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 19)
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Simulated Annealing (Dataset 19)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for LS + SA (Dataset 19)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='LS + SA', linestyle='-', color='cyan')

# Plot for Omega-w + SA (Dataset 19)
plt.plot(time_points_omega_sa, mean_objective_omega_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 19)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()