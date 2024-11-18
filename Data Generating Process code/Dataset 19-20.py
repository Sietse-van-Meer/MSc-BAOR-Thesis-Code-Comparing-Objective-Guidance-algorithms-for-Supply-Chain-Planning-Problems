import random

def generate_jobs(num_jobs, due_dates, p_range, l_j, u_j):
    jobs = []
    for job_id in range(1, num_jobs + 1):
        p_i = random.randint(*p_range)  # Processing time between specified range
        w_i = p_i  # Weight between specified range
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


# Randomized due dates for Dataset 14 (199 unique due dates from 0 to 4599, plus 4600)
due_dates_19 = random.sample(range(0, 1840), 239) + [1840]

# Randomized due dates for Dataset 14 (119 unique due dates from 0 to 4599, plus 4600)
due_dates_20 = random.sample(range(0, 1040), 239) + [1040]


# Generate 500 jobs for each dataset with specified ranges
l_j = 80
u_j = 160
p_range = (40, 160)
w_range = (40, 160)

jobs_dataset_19 = generate_jobs(800, due_dates_19, p_range, l_j, u_j)
jobs_dataset_20 = generate_jobs(800, due_dates_20, p_range, l_j, u_j)

# Display the generated jobs for each dataset
print("Dataset 19 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_19))

print("\nDataset 20 Jobs:")
print(", ".join(str(job) for job in jobs_dataset_20))