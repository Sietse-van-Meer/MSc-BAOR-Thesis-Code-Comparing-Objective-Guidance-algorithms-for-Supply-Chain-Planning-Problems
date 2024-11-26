import matplotlib.pyplot as plt

# Data for each dataset
datasets = {
    "Dataset 1": {
        "time_points_omega_w": [0.009, 0.017, 0.025, 0.031, 0.037],
        "mean_objective_omega_w": [7826.07, 6240.33, 5966.33, 5840.83, 5777.03],
        "time_points_w_omega": [0.009, 0.018, 0.023, 0.031, 0.042],
        "mean_objective_w_omega": [6610.10, 5860.67, 5859.77, 5768.90, 5682.67],
    },
    "Dataset 2": {
        "time_points_omega_w": [0.009, 0.013, 0.023, 0.027, 0.036],
        "mean_objective_omega_w": [270.20, 270.20, 232.67, 232.67, 227.20],
        "time_points_w_omega": [0.005, 0.017, 0.020, 0.036, 0.043],
        "mean_objective_w_omega": [809.87, 237.93, 237.93, 225.50, 218.97],
    },
    "Dataset 4": {
        "time_points_omega_w": [0.007, 0.014, 0.020, 0.025, 0.030],
        "mean_objective_omega_w": [33259.07, 31454.67, 30984.13, 30970.50, 30890.63],
        "time_points_w_omega": [0.007, 0.014, 0.019, 0.025, 0.035],
        "mean_objective_w_omega": [31538.30, 31005.80, 30990.67, 30909.17, 30841.23],
    },
    "Dataset 7": {
        "time_points_omega_w": [0.064, 0.089, 0.165, 0.170, 0.235],
        "mean_objective_omega_w": [229.23, 229.23, 200.17, 200.17, 184.60],
        "time_points_w_omega": [0.027, 0.092, 0.113, 0.175, 0.265],
        "mean_objective_w_omega": [2749.93, 215.90, 215.90, 189.47, 177.33],
    },
    "Dataset 9": {
        "time_points_omega_w": [0.064, 0.094, 0.152, 0.179, 0.236],
        "mean_objective_omega_w": [547456.93, 545288.10, 543358.17, 543288.47, 542926.13],
        "time_points_w_omega": [0.038, 0.106, 0.127, 0.198, 0.280],
        "mean_objective_w_omega": [546385.27, 543706.80, 543692.37, 543255.00, 543186.20],
    },
    "Dataset 10": {
        "time_points_omega_w": [0.108, 0.127, 0.216, 0.232, 0.333],
        "mean_objective_omega_w": [485.97, 485.97, 480.83, 480.83, 477.60],
        "time_points_w_omega": [0.020, 0.145, 0.143, 0.241, 0.337],
        "mean_objective_w_omega": [3236.77, 500.87, 500.87, 490.80, 483.97],
    },
}

# Plotting all datasets with 2 rows and 3 columns layout
# Updating the plot to use the small letter omega (ω) for labels
plt.figure(figsize=(18, 10))

for idx, (dataset, data) in enumerate(datasets.items(), start=1):
    plt.subplot(2, 3, idx)
    plt.plot(data["time_points_omega_w"], data["mean_objective_omega_w"], marker='o', linestyle='-', label='ωwωwω', color='blue')
    plt.plot(data["time_points_w_omega"], data["mean_objective_w_omega"], marker='o', linestyle='-', label='wωwωwω', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Objective Value')
    plt.title(f'Mean Objective Value vs Time ({dataset})')
    plt.grid(True)
    plt.legend()

# Adjusting layout to add more vertical space between rows
plt.subplots_adjust(hspace=0.3)
plt.show()