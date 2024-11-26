import matplotlib.pyplot as plt

# Data for each dataset (duration and mean objective values)
duration = {
    'dataset_1': [0.008, 0.018, 0.026, 0.032, 0.039, 0.044, 0.052],
    'dataset_2': [0.009, 0.013, 0.023, 0.027, 0.037, 0.041, 0.051],
    'dataset_4': [0.007, 0.015, 0.021, 0.025, 0.032, 0.036, 0.043],
    'dataset_7': [0.070, 0.085, 0.146, 0.167, 0.230, 0.257, 0.319],
    'dataset_9': [0.068, 0.107, 0.154, 0.184, 0.245, 0.268, 0.323],
    'dataset_10': [0.106, 0.127, 0.211, 0.230, 0.327, 0.347, 0.441]
}

objective_values = {
    'dataset_1': [7826.07, 6240.33, 5966.33, 5840.83, 5777.03, 5728.60, 5680.83],
    'dataset_2': [270.20, 270.20, 232.67, 232.67, 227.20, 227.20, 220.20],
    'dataset_4': [33259.07, 31454.67, 30984.13, 30970.50, 30890.63, 30890.63, 30845.97],
    'dataset_7': [229.23, 229.23, 200.17, 200.17, 184.60, 184.60, 174.83],
    'dataset_9': [547456.93, 545288.10, 543358.17, 543288.47, 542926.13, 542869.83, 542601.60],
    'dataset_10': [485.97, 485.97, 480.83, 480.83, 477.60, 477.60, 474.07]
}

# Titles for each dataset
titles = ['Dataset 1', 'Dataset 2', 'Dataset 4', 'Dataset 7', 'Dataset 9', 'Dataset 10']

# Create a figure to plot six graphs
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# Plot each dataset
for i, dataset in enumerate(duration.keys()):
    ax = axs[i // 3, i % 3]
    
    # Plot the line for the dataset
    ax.plot(duration[dataset], objective_values[dataset], marker='o', linestyle='-', color='blue')
    
    # Highlight the fifth point
    ax.plot(duration[dataset][4], objective_values[dataset][4], marker='o', markersize=10, color='red', label='Highlighted Point')
    
    # Set titles and labels
    ax.set_title(titles[i])
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Mean Objective Value')

# Adjust layout for better visibility
plt.tight_layout(pad=3.0)
plt.show()