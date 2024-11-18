import matplotlib.pyplot as plt

# Data for RFILS (Dataset 4)
time_points_rfi = [0.007, 0.013, 0.021, 0.027, 0.031]
mean_objective_rfi_new = [33250.47, 33249.27, 33249.10, 33249.10, 33249.10]

# Data for Omega-W (Dataset 4)
time_points_omega_w_new = [0.007, 0.015, 0.021, 0.026, 0.030]
mean_objective_omega_w_dataset4 = [33259.07, 31446.00, 30974.23, 30974.23, 30917.33]

# Data for Tabu Search (Dataset 4)
time_points_tabu_search = [0.007, 0.011, 0.019, 0.023, 0.030]
mean_objective_tabu_search = [32953.13, 32765.40, 32560.33, 32516.30, 32286.07]

# Data for Simulated Annealing (Dataset 4)
time_points_sim_annealing = [0.006, 0.014, 0.020, 0.025, 0.030]
mean_objective_sim_annealing = [33262.00, 30851.63, 30653.63, 30644.40, 30637.57]

# Data for GLS-Q (Dataset 4)
time_points_gls_4 = [0.006, 0.011, 0.017, 0.023, 0.030]
mean_objective_gls_4 = [33273.20, 33081.27, 31974.77, 31327.50, 31050.13]

# Plotting the graph for Dataset 4
plt.figure(figsize=(10, 6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 4)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset4, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 4)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 4)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 4)
plt.plot(time_points_gls_4, mean_objective_gls_4, marker='o', label='GLS-Q', linestyle='-', color='orange')

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