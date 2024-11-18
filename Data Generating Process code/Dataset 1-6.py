import random

def generate_jobs(num_jobs, due_dates):
    jobs = []
    l_j = 10  # Lower bound for all jobs
    u_j = 20  # Upper bound for all jobs
    for job_id in range(1, num_jobs + 1):
        p_i = random.randint(5, 20)  # Processing time between 1 and 4
        w_i = random.randint(5, 20)  # Weight between 1 and 4
        d_i = random.choice(due_dates)  # Due date from the specified set

        # Calculate valid x_i based on the constraints
        min_x_i = max(0, d_i - l_j - p_i)
        max_x_i = d_i + u_j - p_i

        # Ensure min_x_i is not greater than max_x_i, then set x_i
        if min_x_i > max_x_i:
            x_i = min_x_i
        else:
            x_i = random.randint(min_x_i, max_x_i)  # Start time within the valid range

        jobs.append((job_id, p_i, w_i, d_i, x_i, l_j, u_j))
    return jobs

# Randomized due dates for Dataset 1 (14 unique due dates from 0 to 45, plus 46)
due_dates_1 = random.sample(range(0, 230), 22) + [230]

# Randomized due dates for Dataset 2 (19 unique due dates from 0 to 45, plus 46)
due_dates_2 = random.sample(range(0, 230), 29) + [230]

# Randomized due dates for Dataset 3 (24 unique due dates from 0 to 45, plus 46)
due_dates_3 = random.sample(range(0, 230), 37) + [230]

# Randomized due dates for Dataset 4 (14 unique due dates from 0 to 25, plus 26)
due_dates_4 = random.sample(range(0, 130), 22) + [130]

# Randomized due dates for Dataset 5 (19 unique due dates from 0 to 25, plus 26)
due_dates_5 = random.sample(range(0, 130), 29) + [130]

# Randomized due dates for Dataset 6 (24 unique due dates from 0 to 25, plus 26)
due_dates_6 = random.sample(range(0, 130), 37) + [130]

# Generate 100 jobs for each dataset
jobs_dataset_1 = generate_jobs(100, due_dates_1)
jobs_dataset_2 = generate_jobs(100, due_dates_2)
jobs_dataset_3 = generate_jobs(100, due_dates_3)
jobs_dataset_4 = generate_jobs(100, due_dates_4)
jobs_dataset_5 = generate_jobs(100, due_dates_5)
jobs_dataset_6 = generate_jobs(100, due_dates_6)

# Display the generated jobs for each dataset
print("Dataset 1 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_1))

print("\nDataset 2 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_2))

print("\nDataset 3 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_3))

print("\nDataset 4 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_4))

print("\nDataset 5 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_5))

print("\nDataset 6 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_6))