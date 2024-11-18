import matplotlib.pyplot as plt
import numpy as np

# Data for the plots
duration_1 = [0.008, 0.017, 0.024, 0.030, 0.035]
durations_2 = [0.009, 0.013, 0.023, 0.029, 0.039]
durations_4 = [0.009, 0.016, 0.023, 0.027, 0.033]
durations_7 = [0.063, 0.084, 0.140, 0.165, 0.224]
durations_9 = [0.073, 0.105, 0.167, 0.181, 0.237]
durations_10 = [0.112, 0.128, 0.218, 0.237, 0.322]

# Dataset 1
mean_1_obj_nu_05 = [7826.07, 6262.77, 5943.03, 5796.13, 5699.50]
mean_1_obj_nu_0005 = [7826.07, 7063.10, 7006.30, 6966.87, 6929.37]

# Dataset 2
objectives_nu_01_2 = [270.20, 270.20, 222.03, 222.03, 208.63]
objectives_nu_10_2 = [270.20, 270.20, 233.30, 233.30, 228.23]
objectives_nu_05_2 = [270.20, 270.20, 232.23, 232.23, 224.07]

# Dataset 4
mean_obj_nu_05_4 = [33259.07, 31446.00, 30974.23, 30974.23, 30917.33]
mean_obj_nu_005_4 = [33259.07, 31486.53, 31241.93, 31190.77, 31128.90]
mean_obj_nu_05_alt_4 = [33259.07, 31391.97, 30952.77, 30952.63, 30854.43]

# Dataset 7
mean_obj_nu_05_7 = [229.23, 229.23, 196.40, 196.40, 182.13]
mean_obj_nu_005_7 = [229.23, 229.23, 183.97, 183.97, 168.07]
mean_obj_nu_1_7 = [229.23, 229.23, 200.17, 200.17, 184.60]

# Dataset 9
mean_obj_nu_05_9 = [547456.93, 545289.53, 543356.27, 543284.20, 542900.83]
mean_obj_nu_005_9 = [547456.93, 545410.57, 543798.87, 543607.00, 543140.43]
mean_obj_nu_005_alt_9 = [547456.93, 545439.13, 543638.70, 543381.77, 542595.00]

# Dataset 10
mean_obj_nu_05_10 = [485.97, 485.97, 478.50, 478.50, 475.97]
mean_obj_nu_001_10 = [485.97, 485.97, 474.97, 474.97, 467.67]
mean_obj_nu_5_10 = [485.97, 485.97, 481.67, 481.67, 478.50]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Dataset 1
axs[0, 0].plot(duration_1, mean_1_obj_nu_05, marker='o', color='red', label='nu = 0.5')
axs[0, 0].plot(duration_1, mean_1_obj_nu_0005, marker='o', color='blue', label='nu = 0.005')
axs[0, 0].fill_between(duration_1, mean_1_obj_nu_05, mean_1_obj_nu_0005, color='lightblue', alpha=0.5)
axs[0, 0].set_title('Dataset 1')
axs[0, 0].set_xlabel('Duration (seconds)')
axs[0, 0].set_ylabel('Mean Objective Value')
axs[0, 0].legend()

# Dataset 2
axs[0, 1].plot(durations_2, objectives_nu_01_2, color='green', label='nu = 0.01', marker='o')
axs[0, 1].plot(durations_2, objectives_nu_10_2, color='blue', label='nu = 10', marker='o')
axs[0, 1].plot(durations_2, objectives_nu_05_2, color='red', label='nu = 0.5', marker='o')
axs[0, 1].fill_between(durations_2, objectives_nu_01_2, objectives_nu_10_2, color='lightblue', alpha=0.5)
axs[0, 1].set_title('Dataset 2')
axs[0, 1].set_xlabel('Duration (seconds)')
axs[0, 1].set_ylabel('Mean Objective Value')
axs[0, 1].legend()

# Dataset 4
axs[0, 2].plot(durations_4, mean_obj_nu_05_alt_4, color='green', label='nu = 0.05', marker='o')
axs[0, 2].plot(durations_4, mean_obj_nu_005_4, color='blue', label='nu = 0.005', marker='o')
axs[0, 2].plot(durations_4, mean_obj_nu_05_4, color='red', label='nu = 0.5', marker='o')
axs[0, 2].fill_between(durations_4, mean_obj_nu_05_alt_4, mean_obj_nu_005_4, color='lightblue', alpha=0.5)
axs[0, 2].set_title('Dataset 4')
axs[0, 2].set_xlabel('Duration (seconds)')
axs[0, 2].set_ylabel('Mean Objective Value')
axs[0, 2].legend()

# Dataset 7
axs[1, 0].plot(durations_7, mean_obj_nu_05_7, color='red', label='nu = 0.5', marker='o')
axs[1, 0].plot(durations_7, mean_obj_nu_005_7, color='green', label='nu = 0.005', marker='o')
axs[1, 0].plot(durations_7, mean_obj_nu_1_7, color='blue', label='nu = 1', marker='o')
axs[1, 0].fill_between(durations_7, mean_obj_nu_005_7, mean_obj_nu_1_7, color='lightblue', alpha=0.5)
axs[1, 0].set_title('Dataset 7')
axs[1, 0].set_xlabel('Duration (seconds)')
axs[1, 0].set_ylabel('Mean Objective Value')
axs[1, 0].legend()

# Dataset 9
axs[1, 1].plot(durations_9, mean_obj_nu_05_9, color='red', label='nu = 0.5', marker='o')
axs[1, 1].plot(durations_9, mean_obj_nu_005_9, color='blue', label='nu = 0.005', marker='o')
axs[1, 1].plot(durations_9, mean_obj_nu_005_alt_9, color='green', label='nu = 0.05', marker='o')
axs[1, 1].fill_between(durations_9, mean_obj_nu_005_9, mean_obj_nu_005_alt_9, color='lightblue', alpha=0.5)
axs[1, 1].set_title('Dataset 9')
axs[1, 1].set_xlabel('Duration (seconds)')
axs[1, 1].set_ylabel('Mean Objective Value')
axs[1, 1].legend()




# Dataset 10
axs[1, 2].plot(durations_10, mean_obj_nu_05_10, color='red', label='nu = 0.5', marker='o')
axs[1, 2].plot(durations_10, mean_obj_nu_001_10, color='green', label='nu = 0.01', marker='o')
axs[1, 2].plot(durations_10, mean_obj_nu_5_10, color='blue', label='nu = 5', marker='o')
axs[1, 2].fill_between(durations_10, mean_obj_nu_001_10, mean_obj_nu_5_10, color='lightblue', alpha=0.5)
axs[1, 2].set_title('Dataset 10')
axs[1, 2].set_xlabel('Duration (seconds)')
axs[1, 2].set_ylabel('Mean Objective Value')
axs[1, 2].legend()

# Adjust layout to ensure there's no overlap and everything fits
plt.tight_layout()

plt.subplots_adjust(hspace=0.3)

# Show the entire figure with all six graphs
plt.show()