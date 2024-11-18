import matplotlib.pyplot as plt

# Updated data for omega-w original plateau moves (Duration in seconds)
duration_original = {
    'dataset_1': [0.027, 0.032],
    'dataset_2': [0.025, 0.030],
    'dataset_4': [0.029, 0.035],
    'dataset_7': [0.158, 0.188],
    'dataset_9': [0.158, 0.252],
    'dataset_10': [0.305, 0.320]
}

objective_original = {
    'dataset_1': [7333.53, 7292.63],
    'dataset_2': [267.67, 267.30],
    'dataset_4': [31691.67, 31349.17],
    'dataset_7': [227.50, 227.00],
    'dataset_9': [545357.43, 543424.40],
    'dataset_10': [479.23, 476.30]
}

# Updated data for omega-w augmented plateau moves (Duration in seconds)
duration_augmented = {
    'dataset_1': [0.030, 0.039],
    'dataset_2': [0.028, 0.034],
    'dataset_4': [0.030, 0.042],
    'dataset_7': [0.170, 0.192],
    'dataset_9': [0.214, 0.313],
    'dataset_10': [0.311, 0.329]
}

objective_augmented = {
    'dataset_1': [7333.53, 7292.63],
    'dataset_2': [267.67, 267.30],
    'dataset_4': [31685.07, 31344.53],
    'dataset_7': [227.50, 227.00],
    'dataset_9': [545357.43, 543426.90],
    'dataset_10': [479.23, 476.30]
}

# Plot the graphs
plt.figure(figsize=(14, 10))

datasets = ['dataset_1', 'dataset_2', 'dataset_4', 'dataset_7', 'dataset_9', 'dataset_10']
titles = ['Dataset 1', 'Dataset 2', 'Dataset 4', 'Dataset 7', 'Dataset 9', 'Dataset 10']

for i, dataset in enumerate(datasets):
    plt.subplot(2, 3, i+1)
    plt.plot(duration_original[dataset], objective_original[dataset], marker='o', linestyle='-', color='blue', label='Original')
    plt.plot(duration_augmented[dataset], objective_augmented[dataset], marker='o', linestyle='-', color='green', label='Augmented')
    plt.title(titles[i])
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Mean Objective Value')
    plt.legend()

# Adjust the layout with additional vertical spacing between rows
plt.tight_layout(pad=3.0)  # 'pad' parameter increases the space between rows
plt.show()