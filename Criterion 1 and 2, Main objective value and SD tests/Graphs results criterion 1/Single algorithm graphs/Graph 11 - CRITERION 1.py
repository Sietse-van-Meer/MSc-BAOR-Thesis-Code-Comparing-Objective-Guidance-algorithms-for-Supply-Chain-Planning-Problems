import matplotlib.pyplot as plt

# Updated data for RFILS (replaces RFI) based on Dataset 11 provided results
time_points_rfi = [0.036, 0.073, 0.106, 0.141, 0.185]
mean_objective_rfi_new = [500195.93, 500099.20, 500084.47, 500071.43, 500069.37]

# Data for Omega-W (as provided for Dataset 11)
time_points_omega_w_new = [0.041, 0.062, 0.111, 0.130, 0.178]
mean_objective_omega_w_dataset11 = [500197.70, 499171.20, 496268.00, 496268.00, 495925.97]

# Data for Tabu Search (Dataset 11)
time_points_tabu_search = [0.041, 0.070, 0.113, 0.143, 0.182]
mean_objective_tabu_search = [499308.60, 498827.33, 498241.37, 497925.10, 497695.97]

# Corrected SA data for Dataset 11 based on the new information
time_points_sim_annealing = [0.040, 0.064, 0.091, 0.145, 0.181]
mean_objective_sim_annealing = [496175.83, 495610.17, 495601.23, 495592.30, 495590.17]

# GLS-Q data for Dataset 11
time_points_gls_1 = [0.040, 0.085, 0.119, 0.172, 0.185]
mean_objective_gls_1 = [500202.30, 499783.10, 497861.93, 497194.83, 496588.20]

# Plotting the graph for the new dataset
plt.figure(figsize=(10,6))

# Plot for RFILS (new data, replacing RFI)
plt.plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')

# Plot for Omega-W (dataset 11)
plt.plot(time_points_omega_w_new, mean_objective_omega_w_dataset11, marker='o', label='Omega-w', linestyle='-', color='red')

# Plot for Tabu Search (Dataset 11)
plt.plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')

# Plot for Simulated Annealing (Dataset 11)
plt.plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')

# Plot for GLS-Q (Dataset 11)
plt.plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')

# Adding a reference line for Integer Linear Programming (ILP) value
plt.axhline(y=495422, color='lime', linestyle='--', label='ILP')

# Adding labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('Mean Objective Value')
plt.title('Mean Objective Value vs Time for Various Methods (Dataset 11)')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()