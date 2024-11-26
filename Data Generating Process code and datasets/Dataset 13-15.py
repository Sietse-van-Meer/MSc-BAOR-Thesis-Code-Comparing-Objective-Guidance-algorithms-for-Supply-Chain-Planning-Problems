import random

def generate_jobs(num_jobs, due_dates, p_range, w_range, l_j, u_j):
    jobs = []
    for job_id in range(1, num_jobs + 1):
        p_i = random.randint(*p_range)  # Processing time between specified range
        w_i = random.randint(*w_range)  # Weight between specified range
        d_i = random.choice(due_dates)  # Due date from the specified set

        # Calculate valid x_i based on the constraints
        min_x_i = max(0, d_i - l_j - p_i)
        max_x_i = d_i + u_j - p_i

        # Ensure min_x_i is not greater than max_x_i, then set x_i
        if min_x_i > max_x_i:
            x_i = min_x_i  # This line might not be needed or may cause infeasible jobs; check your logic here
        else:
            x_i = random.randint(min_x_i, max_x_i)  # Start time within the valid range

        # Append job as tuple including l_j and u_j
        jobs.append((job_id, p_i, w_i, d_i, x_i, l_j, u_j))
    return jobs

# Randomized due dates for Dataset 13 (89 unique due dates from 0 to 4599, plus 4600)
due_dates_13 = random.sample(range(0, 920), 89) + [920]

# Randomized due dates for Dataset 14 (1199 unique due dates from 0 to 4599, plus 4600)
due_dates_14 = random.sample(range(0, 920), 119) + [920]

# Randomized due dates for Dataset 15 (1499 unique due dates from 0 to 4599, plus 4600)
due_dates_15 = random.sample(range(0, 920), 149) + [920]

# Randomized due dates for Dataset 13 (89 unique due dates from 0 to 4599, plus 4600)
due_dates_16 = random.sample(range(0, 520), 89) + [520]

# Randomized due dates for Dataset 14 (1199 unique due dates from 0 to 4599, plus 4600)
due_dates_17 = random.sample(range(0, 520), 119) + [520]

# Randomized due dates for Dataset 15 (1499 unique due dates from 0 to 4599, plus 4600)
due_dates_18 = random.sample(range(0, 520), 149) + [520]

# Generate 500 jobs for each dataset with specified ranges
l_j = 40
u_j = 80
p_range = (20, 80)
w_range = (20, 80)

jobs_dataset_13 = generate_jobs(400, due_dates_13, p_range, w_range, l_j, u_j)
jobs_dataset_14 = generate_jobs(400, due_dates_14, p_range, w_range, l_j, u_j)
jobs_dataset_15 = generate_jobs(400, due_dates_15, p_range, w_range, l_j, u_j)
jobs_dataset_16 = generate_jobs(400, due_dates_16, p_range, w_range, l_j, u_j)
jobs_dataset_17 = generate_jobs(400, due_dates_17, p_range, w_range, l_j, u_j)
jobs_dataset_18 = generate_jobs(400, due_dates_18, p_range, w_range, l_j, u_j)

# Display the generated jobs for each dataset
print("Dataset 13 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_13))

print("\nDataset 14 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_14))

print("\nDataset 15 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_15))

# Display the generated jobs for each dataset
print("Dataset 16 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_16))

print("\nDataset 17 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_17))

print("\nDataset 18 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_18))