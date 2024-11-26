import random

def generate_jobs(num_jobs, due_dates):
    jobs = []
    l_j = 20  # Lower bound for all jobs
    u_j = 40  # Upper bound for all jobs
    for job_id in range(1, num_jobs + 1):
        p_i = random.randint(10, 40)  # Processing time between 10 and 40
        w_i = random.randint(10, 40)  # Weight between 10 and 40
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

# Randomized due dates for Dataset 7 (44 unique due dates from 0 to 459, plus 460)
due_dates_7 = random.sample(range(0, 460), 44) + [460]

# Randomized due dates for Dataset 8 (59 unique due dates from 0 to 459, plus 460)
due_dates_8 = random.sample(range(0, 460), 59) + [460]

# Randomized due dates for Dataset 9 (74 unique due dates from 0 to 459, plus 460)
due_dates_9 = random.sample(range(0, 460), 74) + [460]

# Randomized due dates for Dataset 10 (44 unique due dates from 0 to 259, plus 260)
due_dates_10 = random.sample(range(0, 260), 44) + [260]

# Randomized due dates for Dataset 11 (59 unique due dates from 0 to 259, plus 260)
due_dates_11 = random.sample(range(0, 260), 59) + [260]

# Randomized due dates for Dataset 12 (74 unique due dates from 0 to 259, plus 260)
due_dates_12 = random.sample(range(0, 260), 74) + [260]

# Generate 300 jobs for each dataset
jobs_dataset_7 = generate_jobs(200, due_dates_7)
jobs_dataset_8 = generate_jobs(200, due_dates_8)
jobs_dataset_9 = generate_jobs(200, due_dates_9)
jobs_dataset_10 = generate_jobs(200, due_dates_10)
jobs_dataset_11 = generate_jobs(200, due_dates_11)
jobs_dataset_12 = generate_jobs(200, due_dates_12)

# Display the generated jobs for each dataset
print("Dataset 7 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_7))

print("\nDataset 8 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_8))

print("\nDataset 9 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_9))

print("\nDataset 10 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_10))

print("\nDataset 11 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_11))

print("\nDataset 12 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_12))
