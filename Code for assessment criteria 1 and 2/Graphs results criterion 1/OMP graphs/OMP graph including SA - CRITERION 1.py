import matplotlib.pyplot as plt

# Data for Omega-w
time_points_omega_w = [0.094, 0.109, 0.206, 0.227, 0.328]
mean_objective_omega_w = [606771.23, 606771.23, 606769.53, 606769.53, 606768.20]

# Data for RFILS
time_points_rfils = [0.094, 0.141, 0.211, 0.255, 0.329]
mean_objective_rfils = [606764.03, 606754.63, 606749.27, 606746.43, 606743.20]

# Data for Simulated Annealing (SA)
time_points_sa = [0.086, 0.148, 0.210, 0.261, 0.331]
mean_objective_sa = [696543.60, 696543.60, 696543.60, 696543.60, 675254.23]

# Data for Tabu Search (TS)
time_points_ts = [0.090, 0.148, 0.210, 0.281, 0.332]
mean_objective_ts = [606786.37, 606773.63, 606772.00, 606772.00, 606772.00]

# Data for Guided Local Search (GLS-Q)
time_points_glsq = [0.089, 0.144, 0.209, 0.266, 0.339]
mean_objective_glsq = [606777.40, 606771.50, 606770.43, 606770.37, 606770.37]

# Data for RFILS + SA
time_points_ls_sa = [0.090, 0.153, 0.203, 0.258, 0.338]
mean_objective_ls_sa = [606771.87, 606771.37, 606771.23, 606771.23, 606771.23]

# Data for ω-w + SA
time_points_omega_w_sa = [0.090, 0.110, 0.206, 0.249, 0.332]
mean_objective_omega_w_sa = [606771.23, 606771.23, 606769.53, 606769.53, 606769.53]

# Plotting the graph for the OMP dataset
plt.figure(figsize=(10, 6))

# Plot for Omega-w
plt.plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for RFILS
plt.plot(time_points_rfils, mean_objective_rfils, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Simulated Annealing (SA)
plt.plot(time_points_sa, mean_objective_sa, marker='o', label='SA', linestyle='-', color='purple')

# Plot for Tabu Search (TS)
plt.plot(time_points_ts, mean_objective_ts, marker='o', label='TS', linestyle='-', color='green')

# Plot for Guided Local Search (GLS-Q)
plt.plot(time_points_glsq, mean_objective_glsq, marker='o', label='GLS-Q', linestyle='-', color='magenta')

# Plot for LS + SA (make cyan)
plt.plot(time_points_ls_sa, mean_objective_ls_sa, marker='o', label='RFILS + SA', linestyle='-', color='cyan')

# Plot for ω-w + SA (make gold)
plt.plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='ω-w + SA', linestyle='-', color='gold')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=606684, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (OMP Dataset)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()