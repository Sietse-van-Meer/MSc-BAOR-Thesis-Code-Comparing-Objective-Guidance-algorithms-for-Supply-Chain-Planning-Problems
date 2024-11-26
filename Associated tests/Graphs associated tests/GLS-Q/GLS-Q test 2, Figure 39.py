import matplotlib.pyplot as plt

# Data for GLS-Q (optimized parameterization) - Dataset 1, 7, 13
durations_glsq_dataset1 = [0.009, 0.015, 0.022, 0.029, 0.039]
mean_obj_glsq_dataset1 = [7846.20, 7160.33, 6223.00, 5843.60, 5499.80]

durations_glsq_dataset7 = [0.055, 0.105, 0.138, 0.188, 0.230]
mean_obj_glsq_dataset7 = [233.60, 229.33, 228.33, 199.73, 185.73]

durations_glsq_dataset13 = [0.399, 0.593, 0.781, 1.011, 1.465]
mean_obj_glsq_dataset13 = [2408.27, 2404.50, 2395.97, 2379.47, 2348.17]

# Data for Alternative Configuration 1
durations_alt1_dataset1 = [0.009, 0.014, 0.022, 0.029, 0.040]
mean_obj_alt1_dataset1 = [7851.30, 6885.70, 6259.10, 5799.40, 5569.70]

durations_alt1_dataset7 = [0.055, 0.096, 0.131, 0.183, 0.230]
mean_obj_alt1_dataset7 = [233.60, 229.33, 229.23, 201.30, 193.97]

durations_alt1_dataset13 = [0.372, 0.620, 0.798, 1.033, 1.381]
mean_obj_alt1_dataset13 = [2409.87, 2404.50, 2403.53, 2375.00, 2346.27]

# Data for Alternative Configuration 2
durations_alt2_dataset1 = [0.009, 0.015, 0.022, 0.028, 0.039]
mean_obj_alt2_dataset1 = [7837.93, 6909.80, 6073.80, 5787.43, 5501.30]

durations_alt2_dataset7 = [0.054, 0.095, 0.131, 0.183, 0.231]
mean_obj_alt2_dataset7 = [233.60, 229.33, 208.50, 186.50, 181.37]

durations_alt2_dataset13 = [0.360, 0.616, 0.830, 1.072, 1.414]
mean_obj_alt2_dataset13 = [2410.73, 2404.50, 2382.17, 2352.43, 2339.17]

# Plotting the results
plt.figure(figsize=(18, 6))

# Plot for Dataset 1
plt.subplot(1, 3, 1)
plt.plot(durations_glsq_dataset1, mean_obj_glsq_dataset1, color='blue', label='Final GLS-Q configuration', marker='o', linestyle='-')
plt.plot(durations_alt1_dataset1, mean_obj_alt1_dataset1, color='orange', label='Alternative 1', marker='^', linestyle='--')
plt.plot(durations_alt2_dataset1, mean_obj_alt2_dataset1, color='green', label='Alternative 2', marker='s', linestyle=':')
plt.title('Dataset 1: Mean Objective vs Duration')
plt.xlabel('Average Duration (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Plot for Dataset 7
plt.subplot(1, 3, 2)
plt.plot(durations_glsq_dataset7, mean_obj_glsq_dataset7, color='blue', label='Final GLS-Q configuration', marker='o', linestyle='-')
plt.plot(durations_alt1_dataset7, mean_obj_alt1_dataset7, color='orange', label='Alternative 1', marker='^', linestyle='--')
plt.plot(durations_alt2_dataset7, mean_obj_alt2_dataset7, color='green', label='Alternative 2', marker='s', linestyle=':')
plt.title('Dataset 7: Mean Objective vs Duration')
plt.xlabel('Average Duration (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

# Plot for Dataset 13
plt.subplot(1, 3, 3)
plt.plot(durations_glsq_dataset13, mean_obj_glsq_dataset13, color='blue', label='Final GLS-Q configuration', marker='o', linestyle='-')
plt.plot(durations_alt1_dataset13, mean_obj_alt1_dataset13, color='orange', label='Alternative 1', marker='^', linestyle='--')
plt.plot(durations_alt2_dataset13, mean_obj_alt2_dataset13, color='green', label='Alternative 2', marker='s', linestyle=':')
plt.title('Dataset 13: Mean Objective vs Duration')
plt.xlabel('Average Duration (seconds)')
plt.ylabel('Mean Objective Value')
plt.legend()

plt.tight_layout()
plt.show()