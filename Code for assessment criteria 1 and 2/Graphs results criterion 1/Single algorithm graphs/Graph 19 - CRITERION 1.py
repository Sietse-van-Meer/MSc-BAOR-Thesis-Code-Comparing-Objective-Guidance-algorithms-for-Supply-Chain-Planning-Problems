import matplotlib.pyplot as plt

# Updated Data for RFILS (Dataset 19)
time_points_rfi = [3.413, 6.257, 8.228, 10.396, 13.130]
mean_objective_rfi = [7280657.27, 7280595.33, 7280585.40, 7280577.67, 7280575.80]

# Updated Data for Omega-W (Dataset 19)
time_points_omega_w = [3.509, 5.401, 8.318, 9.352, 12.610]
mean_objective_omega_w = [7280654.47, 4343605.63, 4320254.97, 4319965.13, 4316317.03]

# Updated Data for Tabu Search (Dataset 19)
time_points_tabu_search = [3.440, 5.764, 7.937, 10.073, 12.783]
mean_objective_tabu_search = [7232452.43, 7088009.10, 6996187.60, 6905206.03, 6826267.53]

# Updated Data for Simulated Annealing (Dataset 19)
time_points_sim_annealing = [3.486, 5.645, 8.143, 10.666, 13.168]
mean_objective_sim_annealing = [4264611.50, 4264456.50, 4264368.47, 4264317.43, 4264278.13]

# Updated Data for GLS-Q (Dataset 19)
time_points_gls_q = [3.599, 5.818, 7.580, 10.103, 13.002]
mean_objective_gls_q = [7280718.33, 6364223.30, 5394860.57, 4703684.80, 4492369.47]

# Plotting the graph for Dataset 19
plt.figure(figsize=(12, 8))

# Plot for RFILS
plt.plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-W', linestyle='-', color='red')

# Plot for Tabu Search
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q
plt.plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 19)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()