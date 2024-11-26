import matplotlib.pyplot as plt
datasets_ori_aug = {
    "Dataset 1": {
        "time_points_ori": [0.029, 0.044],
        "mean_objective_ori": [6240.333, 5966.333],
        "time_points_aug": [0.034, 0.045],
        "mean_objective_aug": [6240.333, 5966.967],
    },
    "Dataset 2": {
        "time_points_ori": [0.022, 0.038],
        "mean_objective_ori": [270.200, 232.667],
        "time_points_aug": [0.032, 0.046],
        "mean_objective_aug": [270.200, 232.967],
    },
    "Dataset 4": {
        "time_points_ori": [0.024, 0.036],
        "mean_objective_ori": [31454.667, 30984.133],
        "time_points_aug": [0.029, 0.040],
        "mean_objective_aug": [31454.667, 30983.267],
    },
    "Dataset 7": {
        "time_points_ori": [0.135, 0.231],
        "mean_objective_ori": [229.233, 200.167],
        "time_points_aug": [0.191, 0.291],
        "mean_objective_aug": [229.233, 198.467],
    },
    "Dataset 9": {
        "time_points_ori": [0.159, 0.247],
        "mean_objective_ori": [545288.100, 543358.167],
        "time_points_aug": [0.201, 0.300],
        "mean_objective_aug": [545288.100, 543356.433],
    },
    "Dataset 10": {
        "time_points_ori": [0.245, 0.406],
        "mean_objective_ori": [485.967, 480.833],
        "time_points_aug": [0.294, 0.450],
        "mean_objective_aug": [485.967, 479.867],
    },
}

# Plotting all datasets with 2 rows and 3 columns layout
plt.figure(figsize=(18, 10))

for idx, (dataset, data) in enumerate(datasets_ori_aug.items(), start=1):
    plt.subplot(2, 3, idx)
    plt.plot(data["time_points_ori"], data["mean_objective_ori"], marker='o', linestyle='-', label='ORI', color='blue')
    plt.plot(data["time_points_aug"], data["mean_objective_aug"], marker='o', linestyle='-', label='AUG', color='orange')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Objective Value')
    plt.title(f'Mean Objective Value vs Time ({dataset})')
    plt.grid(True)
    plt.legend()

# Adjusting layout to add more vertical space between rows
plt.subplots_adjust(hspace=0.5)
plt.show()