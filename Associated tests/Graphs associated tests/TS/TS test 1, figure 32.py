import matplotlib.pyplot as plt

# Duration and mean objective value data for each dataset (TS Tabu List: jobs)
durations_jobs_1 = [0.007, 0.013, 0.020, 0.026, 0.038]
mean_obj_jobs_1 = [7835.33, 7414.93, 6967.33, 6762.50, 6389.60]

durations_jobs_2 = [0.008, 0.016, 0.022, 0.029, 0.036]
mean_obj_jobs_2 = [278.80, 259.50, 248.73, 236.23, 230.60]

durations_jobs_4 = [0.006, 0.012, 0.019, 0.024, 0.030]
mean_obj_jobs_4 = [33065.53, 32816.83, 32472.13, 32330.63, 32159.93]

durations_jobs_7 = [0.055, 0.107, 0.144, 0.180, 0.228]
mean_obj_jobs_7 = [240.90, 223.97, 214.20, 206.17, 196.63]

durations_jobs_9 = [0.063, 0.124, 0.168, 0.205, 0.255]
mean_obj_jobs_9 = [547265.13, 546129.10, 546115.83, 545846.20, 545540.83]

durations_jobs_10 = [0.099, 0.172, 0.221, 0.276, 0.356]
mean_obj_jobs_10 = [490.50, 477.97, 471.30, 466.90, 464.33]

durations_jobs_13 = [0.357, 0.509, 0.830, 1.152, 1.415]
mean_obj_jobs_13 = [2406.90, 2399.00, 2378.23, 2368.03, 2359.93]

durations_jobs_15 = [0.610, 0.840, 1.296, 1.658, 2.173]
mean_obj_jobs_15 = [1992.47, 1963.27, 1929.40, 1915.60, 1882.77]

durations_jobs_17 = [0.815, 1.144, 1.420, 1.893, 2.492]
mean_obj_jobs_17 = [2564.10, 2522.33, 2510.07, 2498.47, 2490.30]

# Duration and mean objective value data for each dataset (TS Tabu List: moves)
durations_moves_1 = [0.006, 0.013, 0.021, 0.030, 0.038]
mean_obj_moves_1 = [7966.67, 7352.40, 6899.70, 6594.47, 6473.10]

durations_moves_2 = [0.007, 0.014, 0.023, 0.030, 0.036]
mean_obj_moves_2 = [278.13, 260.23, 250.57, 239.87, 232.60]

durations_moves_4 = [0.007, 0.013, 0.021, 0.024, 0.030]
mean_obj_moves_4 = [33065.53, 32816.83, 32472.13, 32330.63, 32183.40]

durations_moves_7 = [0.056, 0.088, 0.133, 0.182, 0.231]
mean_obj_moves_7 = [244.17, 228.20, 212.40, 195.20, 182.37]

durations_moves_9 = [0.061, 0.098, 0.147, 0.190, 0.245]
mean_obj_moves_9 = [546401.13, 545725.33, 545424.43, 544930.50, 544636.67]

durations_moves_10 = [0.097, 0.166, 0.237, 0.288, 0.344]
mean_obj_moves_10 = [493.20, 484.43, 479.27, 476.77, 472.63]

durations_moves_13 = [0.372, 0.515, 0.812, 1.089, 1.404]
mean_obj_moves_13 = [2440.63, 2421.23, 2384.23, 2362.13, 2343.10]

durations_moves_15 = [0.628, 0.882, 1.258, 1.580, 2.192]
mean_obj_moves_15 = [1986.60, 1931.00, 1886.13, 1849.77, 1809.83]

durations_moves_17 = [0.849, 1.143, 1.521, 1.879, 2.331]
mean_obj_moves_17 = [2493.30, 2467.27, 2459.83, 2452.43, 2439.53]

# Create a 1x3 grid of subplots for the first row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot for Dataset 1
axs[0].plot(durations_jobs_1, mean_obj_jobs_1, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[0].plot(durations_moves_1, mean_obj_moves_1, color='red', label='TS Tabu List type: moves', marker='o')
axs[0].set_title('Dataset 1')
axs[0].set_xlabel('Duration (seconds)')
axs[0].set_ylabel('Mean Objective Value')
axs[0].legend()

# Plot for Dataset 2
axs[1].plot(durations_jobs_2, mean_obj_jobs_2, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[1].plot(durations_moves_2, mean_obj_moves_2, color='red', label='TS Tabu List type: moves', marker='o')
axs[1].set_title('Dataset 2')
axs[1].set_xlabel('Duration (seconds)')
axs[1].set_ylabel('Mean Objective Value')
axs[1].legend()

# Plot for Dataset 4
axs[2].plot(durations_jobs_4, mean_obj_jobs_4, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[2].plot(durations_moves_4, mean_obj_moves_4, color='red', label='TS Tabu List type: moves', marker='o')
axs[2].set_title('Dataset 4')
axs[2].set_xlabel('Duration (seconds)')
axs[2].set_ylabel('Mean Objective Value')
axs[2].legend()

# Automatically adjust subplots to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()




# Create a 1x3 grid of subplots for the second row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot for Dataset 7
axs[0].plot(durations_jobs_7, mean_obj_jobs_7, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[0].plot(durations_moves_7, mean_obj_moves_7, color='red', label='TS Tabu List type: moves', marker='o')
axs[0].set_title('Dataset 7')
axs[0].set_xlabel('Duration (seconds)')
axs[0].set_ylabel('Mean Objective Value')
axs[0].legend()

# Plot for Dataset 9
axs[1].plot(durations_jobs_9, mean_obj_jobs_9, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[1].plot(durations_moves_9, mean_obj_moves_9, color='red', label='TS Tabu List type: moves', marker='o')
axs[1].set_title('Dataset 9')
axs[1].set_xlabel('Duration (seconds)')
axs[1].set_ylabel('Mean Objective Value')
axs[1].legend()

# Plot for Dataset 10
axs[2].plot(durations_jobs_10, mean_obj_jobs_10, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[2].plot(durations_moves_10, mean_obj_moves_10, color='red', label='TS Tabu List type: moves', marker='o')
axs[2].set_title('Dataset 10')
axs[2].set_xlabel('Duration (seconds)')
axs[2].set_ylabel('Mean Objective Value')
axs[2].legend()

# Automatically adjust subplots to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()


# Create a 1x3 grid of subplots for the third row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot for Dataset 13
axs[0].plot(durations_jobs_13, mean_obj_jobs_13, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[0].plot(durations_moves_13, mean_obj_moves_13, color='red', label='TS Tabu List type: moves', marker='o')
axs[0].set_title('Dataset 13')
axs[0].set_xlabel('Duration (seconds)')
axs[0].set_ylabel('Mean Objective Value')
axs[0].legend()

# Plot for Dataset 15
axs[1].plot(durations_jobs_15, mean_obj_jobs_15, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[1].plot(durations_moves_15, mean_obj_moves_15, color='red', label='TS Tabu List type: moves', marker='o')
axs[1].set_title('Dataset 15')
axs[1].set_xlabel('Duration (seconds)')
axs[1].set_ylabel('Mean Objective Value')
axs[1].legend()

# Plot for Dataset 17
axs[2].plot(durations_jobs_17, mean_obj_jobs_17, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[2].plot(durations_moves_17, mean_obj_moves_17, color='red', label='TS Tabu List type: moves', marker='o')
axs[2].set_title('Dataset 17')
axs[2].set_xlabel('Duration (seconds)')
axs[2].set_ylabel('Mean Objective Value')
axs[2].legend()

# Automatically adjust subplots to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()


# Create a 1x3 grid of subplots for the first row
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot for Dataset 1
axs[0].plot(durations_jobs_1, mean_obj_jobs_1, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[0].plot(durations_moves_1, mean_obj_moves_1, color='red', label='TS Tabu List type: moves', marker='o')
axs[0].set_title('Dataset 1 (small-size dataset)')
axs[0].set_xlabel('Duration (seconds)')
axs[0].set_ylabel('Mean Objective Value')
axs[0].legend()

# Plot for Dataset 7
axs[1].plot(durations_jobs_7, mean_obj_jobs_7, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[1].plot(durations_moves_7, mean_obj_moves_7, color='red', label='TS Tabu List type: moves', marker='o')
axs[1].set_title('Dataset 7 (medium-size dataset)')
axs[1].set_xlabel('Duration (seconds)')
axs[1].set_ylabel('Mean Objective Value')
axs[1].legend()

# Plot for Dataset 13
axs[2].plot(durations_jobs_13, mean_obj_jobs_13, color='blue', label='TS Tabu List type: jobs', marker='o')
axs[2].plot(durations_moves_13, mean_obj_moves_13, color='red', label='TS Tabu List type: moves', marker='o')
axs[2].set_title('Dataset 13 (large-size dataset)')
axs[2].set_xlabel('Duration (seconds)')
axs[2].set_ylabel('Mean Objective Value')
axs[2].legend()

# Automatically adjust subplots to avoid overlap
plt.tight_layout()

# Show the plot
plt.show()
