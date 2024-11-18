import matplotlib.pyplot as plt

# Data for (Overload)^2 results
duration_overload = {
    'dataset_1': [0.032, 0.042],
    'dataset_2': [0.026, 0.034],
    'dataset_4': [0.030, 0.046],
    'dataset_7': [0.155, 0.169],
    'dataset_9': [0.164, 0.260],
    'dataset_10': [0.316, 0.332]
}

objective_overload = {
    'dataset_1': [7333.53, 7292.63],
    'dataset_2': [267.67, 267.30],
    'dataset_4': [31691.67, 31349.17],
    'dataset_7': [227.50, 227.00],
    'dataset_9': [545357.43, 543424.40],
    'dataset_10': [479.23, 476.30]
}

# Data for (Load)^2 results
duration_load = {
    'dataset_1': [0.031, 0.048],
    'dataset_2': [0.023, 0.043],
    'dataset_4': [0.025, 0.038],
    'dataset_7': [0.137, 0.238],
    'dataset_9': [0.161, 0.256],
    'dataset_10': [0.240, 0.402]
}

objective_load = {
    'dataset_1': [6240.33, 5966.33],
    'dataset_2': [270.20, 232.67],
    'dataset_4': [31454.67, 30984.13],
    'dataset_7': [229.23, 200.17],
    'dataset_9': [545288.10, 543358.17],
    'dataset_10': [485.97, 480.83]
}

# Data for -(Earliness)^2 results
duration_earliness = {
    'dataset_1': [0.023, 0.032],
    'dataset_2': [0.026, 0.036],
    'dataset_4': [0.022, 0.028],
    'dataset_7': [0.148, 0.208],
    'dataset_9': [0.160, 0.189],
    'dataset_10': [0.264, 0.390]
}

objective_earliness = {
    'dataset_1': [7820.33, 7813.40],
    'dataset_2': [269.83, 266.07],
    'dataset_4': [33252.97, 33243.77],
    'dataset_7': [229.23, 222.90],
    'dataset_9': [547456.93, 547451.97],
    'dataset_10': [485.97, 473.63]
}

# Data for (Load)^2 - (Earliness)^2 results
duration_load_earliness = {
    'dataset_1': [0.042, 0.057],
    'dataset_2': [0.025, 0.045],
    'dataset_4': [0.031, 0.043],
    'dataset_7': [0.168, 0.266],
    'dataset_9': [0.203, 0.302],
    'dataset_10': [0.272, 0.442]
}

objective_load_earliness = {
    'dataset_1': [6266.40, 5989.93],
    'dataset_2': [270.20, 231.63],
    'dataset_4': [31470.40, 30996.83],
    'dataset_7': [229.23, 188.37],
    'dataset_9': [545275.03, 543345.40],
    'dataset_10': [485.97, 478.83]
}

# Data for (Load)^2 + (Overload)^2 results
duration_load_overload = {
    'dataset_1': [0.042, 0.057],
    'dataset_2': [0.030, 0.055],
    'dataset_4': [0.040, 0.056],
    'dataset_7': [0.190, 0.301],
    'dataset_9': [0.208, 0.314],
    'dataset_10': [0.315, 0.513]
}

objective_load_overload = {
    'dataset_1': [6325.23, 6001.10],
    'dataset_2': [270.20, 231.57],
    'dataset_4': [31449.13, 30976.43],
    'dataset_7': [229.23, 190.70],
    'dataset_9': [545358.10, 543422.80],
    'dataset_10': [485.97, 477.50]
}

# Create a figure to plot six graphs
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

datasets = ['dataset_1', 'dataset_2', 'dataset_4', 'dataset_7', 'dataset_9', 'dataset_10']
titles = ['Dataset 1', 'Dataset 2', 'Dataset 4', 'Dataset 7', 'Dataset 9', 'Dataset 10']

# Plot each dataset in a different graph, showing all five methods
for i, dataset in enumerate(datasets):
    ax = axs[i // 3, i % 3]
    
    # Plot (Overload)^2 results
    ax.plot(duration_overload[dataset], objective_overload[dataset], marker='o', linestyle='-', color='blue', label='(Overload)^2')
    
    # Plot (Load)^2 results
    ax.plot(duration_load[dataset], objective_load[dataset], marker='o', linestyle='-', color='green', label='(Load)^2')
    
    # Plot -(Earliness)^2 results
    ax.plot(duration_earliness[dataset], objective_earliness[dataset], marker='o', linestyle='-', color='red', label='-(Earliness)^2')
    
    # Plot (Load)^2 - (Earliness)^2 results
    ax.plot(duration_load_earliness[dataset], objective_load_earliness[dataset], marker='o', linestyle='-', color='purple', label='(Load)^2 - (Earliness)^2')
    
    # Plot (Load)^2 + (Overload)^2 results
    ax.plot(duration_load_overload[dataset], objective_load_overload[dataset], marker='o', linestyle='-', color='orange', label='(Load)^2 + (Overload)^2')
    
    ax.set_title(titles[i])
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Mean Objective Value')
    ax.legend()

# Adjust layout for better visibility
plt.tight_layout(pad=3.0)
plt.show()