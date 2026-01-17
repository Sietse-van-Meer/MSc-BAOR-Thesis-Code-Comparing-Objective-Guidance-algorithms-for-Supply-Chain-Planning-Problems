from numba import njit
import random
import numpy as np
import time

@njit(cache=True)
def calculate_load_at_time(jobs, max_time):
    load_at_time = np.zeros(max_time + 1, dtype=np.int32)
    for job in jobs:
        start, duration, weight = job['start_time'], job['processing_time'], job['weight']
        load_at_time[start:start + duration] += weight
    return load_at_time

@njit(cache=True)
def adjust_load(load_at_time, job, new_start, old_start):
    duration = job['processing_time']
    weight = job['weight']
    if new_start != old_start:
        load_at_time[old_start:old_start + duration] -= weight
        load_at_time[new_start:new_start + duration] += weight

@njit(cache=True)
def calculate_total_overload(load_at_time, c_t):
    overload = np.sum(np.maximum(0, load_at_time - c_t))
    return overload

@njit(cache=True)
def calculate_tardiness(jobs):
    tardiness = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in jobs], dtype=np.int32)
    return tardiness, np.sum(tardiness)

@njit(cache=True)
def calculate_objective_value(total_overload, total_tardiness):
    return 10 * total_overload + total_tardiness

@njit(cache=True)
def is_within_constraints(job, new_start):
    job_end = new_start + job['processing_time']
    lower_bound = job['due_date'] - job['lower_bound']
    upper_bound = job['due_date'] + job['upper_bound']
    return lower_bound <= job_end <= upper_bound

@njit(cache=True)
def evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness):
    duration = job['processing_time']
    weight = job['weight']
    max_time = len(load_at_time) - 1

    # Temporarily adjust load for the new start
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] += weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] -= weight

    # Calculate the new overload and objective
    new_overload = calculate_total_overload(load_at_time, c_t)
    new_tardiness = max(0, new_start + duration - job['due_date'])
    delta_tardiness = new_tardiness - max(0, old_start + duration - job['due_date'])
    new_objective = calculate_objective_value(new_overload, total_tardiness + delta_tardiness)

    # Revert the temporary adjustment
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] -= weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] += weight

    return new_objective, new_overload, delta_tardiness


@njit(cache=True)
def simulated_annealing(jobs, initial_temp, beta, max_time, c_t, num_iterations, seed_value):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    load_at_time = calculate_load_at_time(current_solution, max_time)
    total_overload = calculate_total_overload(load_at_time, c_t)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    total_tardiness = np.sum(tardiness_values)
    current_objective = calculate_objective_value(total_overload, total_tardiness)
    best_objective = current_objective

    current_temp = initial_temp
    iterations = 0
    shifts = np.arange(-10, 11)  # Shifts range from -20 to 20 inclusive
    shifts = shifts[shifts != 0]  # Remove the zero to avoid no-op moves

    while iterations < num_iterations:
        job_index = np.random.randint(0, len(jobs) - 1)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(shifts)
        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            new_objective, new_overload, delta_tardiness = evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness)

            if new_objective <= current_objective or np.random.random() < np.exp((current_objective - new_objective) / current_temp):
                # Accept the move
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] = max(0, new_start + job['processing_time'] - job['due_date'])
                total_tardiness = np.sum(tardiness_values)
                total_overload = new_overload
                current_objective = new_objective
                if new_objective < best_objective:
                    best_objective = new_objective

        # Logarithmic cooling schedule
        current_temp = initial_temp / (1 + beta * np.log(1 + iterations))
        
        iterations += 1

    return best_objective  # Return the best objective found after all iterations



def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                    ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                    ('upper_bound', np.int32)])
    jobs = np.array([(1, 12, 10, 14, 20, 10, 20), (2, 12, 12, 112, 95, 10, 20), (3, 12, 17, 221, 217, 10, 20), (4, 9, 8, 203, 214, 10, 20), (5, 11, 6, 11, 17, 10, 20), (6, 9, 7, 129, 118, 10, 20), (7, 6, 15, 85, 87, 10, 20), (8, 14, 14, 227, 230, 10, 20), (9, 20, 16, 14, 12, 10, 20), (10, 10, 8, 15, 25, 10, 20), (11, 10, 19, 79, 72, 10, 20), (12, 19, 12, 79, 73, 10, 20), (13, 5, 10, 114, 127, 10, 20), (14, 12, 16, 32, 12, 10, 20), (15, 7, 15, 21, 23, 10, 20), (16, 5, 18, 79, 82, 10, 20), (17, 15, 13, 125, 117, 10, 20), (18, 13, 14, 135, 134, 10, 20), (19, 19, 10, 118, 92, 10, 20), (20, 13, 11, 135, 134, 10, 20), (21, 13, 15, 174, 178, 10, 20), (22, 9, 9, 174, 168, 10, 20), (23, 11, 17, 192, 175, 10, 20), (24, 18, 19, 11, 3, 10, 20), (25, 13, 15, 114, 105, 10, 20), (26, 19, 5, 32, 9, 10, 20), (27, 7, 19, 79, 63, 10, 20), (28, 20, 15, 135, 114, 10, 20), (29, 14, 19, 128, 107, 10, 20), (30, 6, 16, 34, 19, 10, 20), (31, 13, 13, 221, 217, 10, 20), (32, 14, 17, 192, 187, 10, 20), (33, 11, 7, 34, 23, 10, 20), (34, 5, 5, 125, 123, 10, 20), (35, 19, 15, 32, 26, 10, 20), (36, 15, 6, 66, 58, 10, 20), (37, 20, 7, 11, 10, 10, 20), (38, 6, 8, 66, 67, 10, 20), (39, 6, 9, 90, 92, 10, 20), (40, 14, 13, 196, 182, 10, 20), (41, 17, 17, 128, 120, 10, 20), (42, 17, 6, 112, 102, 10, 20), (43, 5, 20, 11, 14, 10, 20), (44, 8, 9, 196, 186, 10, 20), (45, 7, 9, 118, 114, 10, 20), (46, 13, 8, 32, 21, 10, 20), (47, 19, 16, 54, 31, 10, 20), (48, 18, 5, 14, 1, 10, 20), (49, 7, 8, 34, 41, 10, 20), (50, 8, 13, 79, 86, 10, 20), (51, 5, 10, 118, 116, 10, 20), (52, 14, 16, 112, 94, 10, 20), (53, 11, 19, 128, 110, 10, 20), (54, 20, 17, 118, 94, 10, 20), (55, 16, 16, 227, 223, 10, 20), (56, 7, 12, 135, 142, 10, 20), (57, 20, 9, 227, 205, 10, 20), (58, 18, 13, 230, 218, 10, 20), (59, 8, 15, 227, 212, 10, 20), (60, 11, 17, 85, 86, 10, 20), (61, 20, 14, 196, 193, 10, 20), (62, 19, 12, 230, 206, 10, 20), (63, 14, 10, 118, 101, 10, 20), (64, 5, 15, 46, 33, 10, 20), (65, 20, 17, 15, 9, 10, 20), (66, 20, 17, 11, 8, 10, 20), (67, 6, 18, 15, 20, 10, 20), (68, 19, 14, 85, 82, 10, 20), (69, 6, 16, 174, 182, 10, 20), (70, 6, 13, 118, 118, 10, 20), (71, 8, 10, 11, 2, 10, 20), (72, 14, 7, 183, 175, 10, 20), (73, 9, 11, 54, 52, 10, 20), (74, 19, 11, 125, 101, 10, 20), (75, 13, 11, 114, 95, 10, 20), (76, 16, 10, 114, 93, 10, 20), (77, 8, 20, 118, 123, 10, 20), (78, 12, 8, 54, 58, 10, 20), (79, 20, 9, 227, 219, 10, 20), (80, 9, 19, 129, 115, 10, 20), (81, 7, 10, 118, 116, 10, 20), (82, 13, 11, 129, 113, 10, 20), (83, 11, 19, 54, 37, 10, 20), (84, 5, 5, 15, 16, 10, 20), (85, 7, 20, 114, 115, 10, 20), (86, 11, 14, 66, 55, 10, 20), (87, 19, 11, 178, 162, 10, 20), (88, 16, 6, 11, 10, 10, 20), (89, 14, 10, 32, 14, 10, 20), (90, 14, 7, 192, 189, 10, 20), (91, 12, 10, 178, 174, 10, 20), (92, 17, 5, 227, 207, 10, 20), (93, 11, 15, 54, 56, 10, 20), (94, 16, 16, 196, 199, 10, 20), (95, 14, 11, 15, 10, 10, 20), (96, 14, 11, 135, 111, 10, 20), (97, 16, 10, 178, 156, 10, 20), (98, 10, 9, 54, 55, 10, 20), (99, 7, 11, 79, 90, 10, 20), (100, 9, 20, 54, 52, 10, 20)
    ], dtype=Job)

    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]

    c_t = 100
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    num_iterations = 146000
    cooling_rates = [0.001, 0.01, 0.1, 0.2, 0.5, 0.9]  # Cooling rates to test

    # Step 1: Set initial temperature as len(jobs) * 10
    initial_temp = len(jobs) * 10

# Step 2: Test different cooling rates using the fixed initial temperature
    best_overall_objective = float('inf')
    best_cooling_rate = None

    for cooling_rate in cooling_rates:
        objective_values = []
        durations_list = []

        for i in range(32):
            seed_value = seeds[i]

            start_time = time.time()

            # Run the simulated annealing algorithm
            current_objective = simulated_annealing(
                jobs, initial_temp, cooling_rate, max_time, c_t, num_iterations, seed_value
            )
            duration = time.time() - start_time

            if i > 1:  # Exclude the first two runs for JIT compilation
                objective_values.append(current_objective)
                durations_list.append(duration)

        # Calculate summary statistics for this cooling rate
        mean_objective = np.mean(objective_values)
        std_dev_objective = np.std(objective_values)
        min_objective = np.min(objective_values)
        max_objective = np.max(objective_values)
        q1_objective = np.percentile(objective_values, 25)
        q3_objective = np.percentile(objective_values, 75)
        average_duration = np.mean(durations_list)

        # Print summary for the current cooling rate
        print(f"\n Cooling Rate = {cooling_rate}:")
        print(f"Mean Objective Value: {mean_objective:.2f}")
        print(f"Standard Deviation: {std_dev_objective:.2f}")
        print(f"Minimum Objective Value: {min_objective:.2f}")
        print(f"Maximum Objective Value: {max_objective:.2f}")
        print(f"Q1 Objective (25th percentile): {q1_objective:.2f}")
        print(f"Q3 Objective (75th percentile): {q3_objective:.2f}")
        print(f"Average Duration: {average_duration:.3f} seconds")

        if mean_objective < best_overall_objective:
            best_overall_objective = mean_objective
            best_cooling_rate = cooling_rate

    print(f"Best Overall Mean Objective Value: {best_overall_objective}")
        
if __name__ == "__main__":
    main()
