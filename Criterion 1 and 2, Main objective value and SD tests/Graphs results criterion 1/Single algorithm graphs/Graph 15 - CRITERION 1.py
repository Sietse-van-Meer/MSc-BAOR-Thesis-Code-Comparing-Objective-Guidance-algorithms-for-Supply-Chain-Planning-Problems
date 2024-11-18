import matplotlib.pyplot as plt

# Data for RFILS (Dataset 15)
time_points_rfi = [0.549, 1.229, 1.568, 1.875, 2.201]
mean_objective_rfi_new = [2015.77, 1998.80, 1995.80, 1994.00, 1993.90]

# Data for Omega-W (Dataset 15)
time_points_omega_w_new = [0.650, 0.779, 1.386, 1.535, 2.161]
mean_objective_omega_w_dataset15 = [2009.57, 2009.57, 1867.60, 1867.60, 1822.10]

# Data for Tabu Search (Dataset 15)
time_points_tabu_search = [0.637, 1.007, 1.410, 1.714, 2.273]
mean_objective_tabu_search = [2031.53, 1990.60, 1969.43, 1947.73, 1923.57]

# Corrected SA data for Dataset 15
time_points_sim_annealing = [0.621, 1.068, 1.467, 1.818, 2.197]
mean_objective_sim_annealing = [1739.40, 1657.83, 1630.13, 1611.63, 1596.70]

# GLS-Q data for Dataset 15
time_points_gls_1 = [0.610, 0.914, 1.362, 1.723, 2.282]
mean_objective_gls_1 = [2029.60, 2015.30, 1983.33, 1938.93, 1851.53]

# Plotting the graph for Dataset 15
plt.figure(figsize=(10,6))

# Plot for RFILS (Dataset 15)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (Dataset 15)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset15, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 15)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 15)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 15)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 15)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()