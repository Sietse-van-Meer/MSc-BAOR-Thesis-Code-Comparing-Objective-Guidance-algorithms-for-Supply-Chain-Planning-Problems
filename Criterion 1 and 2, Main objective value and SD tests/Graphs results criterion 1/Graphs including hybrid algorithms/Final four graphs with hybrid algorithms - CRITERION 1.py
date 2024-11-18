import matplotlib.pyplot as plt

# Creating a figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Dataset 1
time_points_rfi = [0.007, 0.017, 0.024, 0.030, 0.037]
mean_objective_rfi_new = [7828.43, 7800.57, 7800.37, 7800.33, 7800.20]
time_points_omega_w_new = [0.009, 0.017, 0.025, 0.031, 0.037]
mean_objective_omega_w_dataset1 = [7826.07, 6262.77, 5943.03, 5796.13, 5699.50]
time_points_sim_annealing = [0.007, 0.015, 0.020, 0.029, 0.039]
mean_objective_sim_annealing = [8155.87, 5055.13, 4944.23, 4930.87, 4866.53]
time_points_rfils_sa = [0.009, 0.013, 0.022, 0.026, 0.037]
mean_objective_rfils_sa = [7829.03, 7424.80, 5298.57, 5029.63, 4963.97]
time_points_omega_sa = [0.008, 0.017, 0.026, 0.034, 0.037]
mean_objective_omega_sa = [7826.07, 6262.77, 5943.03, 5943.03, 5244.90]

# Plot for Dataset 1
axs[0, 0].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[0, 0].plot(time_points_omega_w_new, mean_objective_omega_w_dataset1, marker='o', label='Omega-w', linestyle='-', color='red')
axs[0, 0].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[0, 0].plot(time_points_rfils_sa, mean_objective_rfils_sa, marker='o', label='RFILS + SA', linestyle='-', color='cyan')
axs[0, 0].plot(time_points_omega_sa, mean_objective_omega_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')
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
time_points_sim_annealing = [0.054, 0.085, 0.143, 0.186, 0.225]
mean_objective_sim_annealing = [304.73, 213.47, 196.60, 190.03, 186.03]
time_points_rfils_sa = [0.060, 0.094, 0.120, 0.166, 0.231]
mean_objective_rfils_sa = [229.60, 229.23, 189.73, 168.13, 159.77]
time_points_omega_w_sa7 = [0.058, 0.082, 0.145, 0.194, 0.234]
mean_objective_omega_w_sa_dataset7 = [229.23, 229.23, 196.40, 196.40, 165.57]

# Plot for Dataset 7
axs[0, 1].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[0, 1].plot(time_points_omega_w_new, mean_objective_omega_w_dataset7, marker='o', label='Omega-w', linestyle='-', color='red')
axs[0, 1].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[0, 1].plot(time_points_rfils_sa, mean_objective_rfils_sa, marker='o', label='RFILS + SA', linestyle='-', color='cyan')
axs[0, 1].plot(time_points_omega_w_sa7, mean_objective_omega_w_sa_dataset7, marker='o', label='Omega-w + SA', linestyle='-', color='gold')
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
time_points_sim_annealing = [0.371, 0.565, 0.795, 1.058, 1.423]
mean_objective_sim_annealing = [2701.77, 2170.17, 2132.73, 2113.60, 2096.10]
time_points_rfils_sa = [0.372, 0.566, 0.843, 1.038, 1.396]
mean_objective_rfils_sa = [2404.93, 2403.53, 2163.20, 2069.67, 2040.07]
time_points_omega_w_sa13 = [0.539, 0.562, 0.946, 1.133, 1.410]
mean_objective_omega_w_sa_dataset13 = [2403.53, 2403.53, 2362.63, 2362.63, 2135.07]

# Plot for Dataset 13
axs[1, 0].plot(time_points_rfi, mean_objective_rfi_new, marker='o', label='RFILS', linestyle='-', color='blue')
axs[1, 0].plot(time_points_omega_w_new, mean_objective_omega_w_dataset13, marker='o', label='Omega-w', linestyle='-', color='red')
axs[1, 0].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[1, 0].plot(time_points_rfils_sa, mean_objective_rfils_sa, marker='o', label='RFILS + SA', linestyle='-', color='cyan')
axs[1, 0].plot(time_points_omega_w_sa13, mean_objective_omega_w_sa_dataset13, marker='o', label='Omega-w + SA', linestyle='-', color='gold')
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
time_points_sim_annealing = [3.486, 5.645, 8.143, 10.666, 13.168]
mean_objective_sim_annealing = [4264611.50, 4264456.50, 4264368.47, 4264317.43, 4264278.13]
time_points_rfils_sa = [3.268, 5.919, 8.393, 10.897, 12.899]
mean_objective_rfils_sa = [5618376.97, 4268211.17, 4267271.40, 4267166.93, 4267109.50]
time_points_omega_w_sa = [3.885, 5.792, 8.991, 10.293, 12.738]
mean_objective_omega_w_sa = [7280654.47, 4343605.63, 4320254.97, 4185674.00, 4184725.37]

# Plot for Dataset 19
axs[1, 1].plot(time_points_rfi, mean_objective_rfi, marker='o', label='RFILS', linestyle='-', color='blue')
axs[1, 1].plot(time_points_omega_w, mean_objective_omega_w, marker='o', label='Omega-w', linestyle='-', color='red')
axs[1, 1].plot(time_points_sim_annealing, mean_objective_sim_annealing, marker='o', label='SA', linestyle='-', color='purple')
axs[1, 1].plot(time_points_rfils_sa, mean_objective_rfils_sa, marker='o', label='RFILS + SA', linestyle='-', color='cyan')
axs[1, 1].plot(time_points_omega_w_sa, mean_objective_omega_w_sa, marker='o', label='Omega-w + SA', linestyle='-', color='gold')
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