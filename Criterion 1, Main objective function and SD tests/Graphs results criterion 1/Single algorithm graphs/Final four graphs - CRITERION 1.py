import matplotlib.pyplot as plt

# Creating a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Dataset 1
time_points_rfi = [0.007, 0.017, 0.024, 0.030, 0.037]
mean_objective_rfi_new = [7828.43, 7800.57, 7800.37, 7800.33, 7800.20]
time_points_omega_w_new = [0.009, 0.017, 0.025, 0.031, 0.037]
mean_objective_omega_w_dataset1 = [7826.07, 6262.77, 5943.03, 5796.13, 5699.50]
time_points_tabu_search = [0.008, 0.012, 0.017, 0.025, 0.039]
mean_objective_tabu_search = [7636.17, 7352.40, 6981.37, 6726.33, 6472.23]
time_points_sim_annealing = [0.007, 0.015, 0.020, 0.029, 0.039]
mean_objective_sim_annealing = [8155.87, 5055.13, 4944.23, 4930.87, 4866.53]
time_points_gls_1 = [0.009, 0.015, 0.022, 0.029, 0.039]
mean_objective_gls_1 = [7846.20, 7160.33, 6223.00, 5843.60, 5499.80]

# Plot for Dataset 1
axs[0, 0].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[0, 0].plot(time_points_omega_w_new, mean_objective_omega_w_dataset1, marker='o', label='Omega-w', linestyle='-', color='red')
axs[0, 0].plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')
axs[0, 0].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[0, 0].plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')
axs[0, 0].set_title('Dataset 1')
axs[0, 0].set_xlabel('Time (seconds)')
axs[0, 0].set_ylabel('Mean Objective Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Dataset 7
time_points_rfi = [0.054, 0.090, 0.148, 0.184, 0.238]
mean_objective_rfi_new = [229.30, 226.87, 225.47, 225.07, 224.77]
time_points_omega_w_new = [0.060, 0.083, 0.146, 0.161, 0.225]
mean_objective_omega_w_dataset7 = [229.23, 229.23, 196.40, 196.40, 182.13]
time_points_tabu_search = [0.060, 0.089, 0.123, 0.163, 0.226]
mean_objective_tabu_search = [243.50, 228.20, 217.10, 199.63, 182.83]
time_points_sim_annealing = [0.054, 0.085, 0.143, 0.186, 0.225]
mean_objective_sim_annealing = [304.73, 213.47, 196.60, 190.03, 186.03]
time_points_gls_1 = [0.055, 0.105, 0.138, 0.188, 0.230]
mean_objective_gls_1 = [233.60, 229.33, 228.33, 199.73, 185.73]

# Plot for Dataset 7
axs[0, 1].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[0, 1].plot(time_points_omega_w_new, mean_objective_omega_w_dataset7, marker='o', label='Omega-w', linestyle='-', color='red')
axs[0, 1].plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')
axs[0, 1].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[0, 1].plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')
axs[0, 1].set_title('Dataset 7')
axs[0, 1].set_xlabel('Time (seconds)')
axs[0, 1].set_ylabel('Mean Objective Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Dataset 13
time_points_rfi = [0.423, 0.495, 0.906, 1.180, 1.470]
mean_objective_rfi_new = [2405.60, 2401.17, 2398.60, 2397.70, 2397.50]
time_points_omega_w_new = [0.399, 0.526, 0.889, 1.032, 1.379]
mean_objective_omega_w_dataset13 = [2403.53, 2403.53, 2362.63, 2362.63, 2357.87]
time_points_tabu_search = [0.386, 0.585, 0.822, 1.140, 1.388]
mean_objective_tabu_search = [2394.93, 2380.77, 2372.50, 2364.90, 2349.87]
time_points_sim_annealing = [0.371, 0.565, 0.795, 1.058, 1.423]
mean_objective_sim_annealing = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]
time_points_gls_1 = [0.399, 0.593, 0.781, 1.011, 1.465]
mean_objective_gls_1 = [2408.27, 2404.50, 2395.97, 2379.47, 2348.17]

# Plot for Dataset 13
axs[1, 0].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[1, 0].plot(time_points_omega_w_new, mean_objective_omega_w_dataset13, marker='o', label='Omega-w', linestyle='-', color='red')
axs[1, 0].plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')
axs[1, 0].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[1, 0].plot(time_points_gls_1, mean_objective_gls_1, marker='o', label='GLS-Q', linestyle='-', color='orange')
axs[1, 0].set_title('Dataset 13')
axs[1, 0].set_xlabel('Time (seconds)')
axs[1, 0].set_ylabel('Mean Objective Value')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Dataset 19
time_points_rfi = [3.413, 6.257, 8.228, 10.396, 13.130]
mean_objective_rfi = [7280657.27, 7280595.33, 7280585.40, 7280577.67, 7280575.80]
time_points_omega_w = [3.509, 5.401, 8.318, 9.352, 12.610]
mean_objective_omega_w = [7280654.47, 4343605.63, 4320254.97, 4319965.13, 4316317.03]
# Data for Tabu Search (Dataset 19)
time_points_tabu_search = [3.440, 5.764, 7.937, 10.073, 12.783]
mean_objective_tabu_search = [7232452.43, 7088009.10, 6996187.60, 6905206.03, 6826267.53]

# Data for Simulated Annealing (Dataset 19)
time_points_sim_annealing = [3.486, 5.645, 8.143, 10.666, 13.168]
mean_objective_sim_annealing = [4264611.50, 4264456.50, 4264368.47, 4264317.43, 4264278.13]

# Data for GLS-Q (Dataset 19)
time_points_gls_q = [3.599, 5.818, 7.580, 10.103, 13.002]
mean_objective_gls_q = [7280718.33, 6364223.30, 5394860.57, 4703684.80, 4492369.47]

# Plot for Dataset 19
axs[1, 1].plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')
axs[1, 1].plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')
axs[1, 1].plot(time_points_tabu_search, mean_objective_tabu_search, marker='o', label='TS', linestyle='-', color='green')
axs[1, 1].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[1, 1].plot(time_points_gls_q, mean_objective_gls_q, marker='o', label='GLS-Q', linestyle='-', color='orange')
axs[1, 1].set_title('Dataset 19')
axs[1, 1].set_xlabel('Time (seconds)')
axs[1, 1].set_ylabel('Mean Objective Value')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Tight layout to avoid overlap and adjust the spacing between rows
plt.tight_layout()

# Adjust vertical space between the two rows
plt.subplots_adjust(hspace=0.4)

# Show the 2x2 plot with additional vertical spacing
plt.show()