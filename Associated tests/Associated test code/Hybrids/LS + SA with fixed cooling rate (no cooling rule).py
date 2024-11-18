from numba import njit
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
def hybrid_local_search_simulated_annealing(jobs, c_t, max_time, max_iterations, max_plateau_moves, seed_value, initial_temp, cooling_rate):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    load_at_time = calculate_load_at_time(current_solution, max_time)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    total_tardiness = np.sum(tardiness_values)
    total_overload = calculate_total_overload(load_at_time, c_t)
    current_objective = calculate_objective_value(total_overload, total_tardiness)
    best_objective = current_objective

    job_indices = np.arange(len(jobs))
    all_shifts = np.arange(-10, 11)
    plateau_moves = 0  # Initialize counter for plateau moves
    current_temp = initial_temp  # Initial temperature for simulated annealing

    iteration = 0  # Initialize iteration counter

    # Regular local search: Accept equal or better moves
    while plateau_moves < max_plateau_moves and iteration < max_iterations:
        job_index = np.random.choice(job_indices)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(all_shifts)
        
        if shift == 0:
            continue
        
        new_start = old_start + shift
        
        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            new_objective, new_overload, delta_tardiness = evaluate_move(
                load_at_time, job, new_start, old_start, c_t, total_tardiness
            )
            iteration += 1  # Increment iteration count

            # Accept if the move is equal or better
            if new_objective <= current_objective:
                if new_objective < current_objective:
                    plateau_moves = 0  # Reset plateau moves on improvement
                else:
                    plateau_moves += 1  # Increment plateau moves if equal

                # Make the move
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] += delta_tardiness
                total_tardiness += delta_tardiness
                total_overload = new_overload
                current_objective = new_objective
                if new_objective < best_objective:
                    best_objective = new_objective
            else:
                plateau_moves += 1  # Increment plateau moves if no improvement

    # Switch to simulated annealing phase after hitting plateau_limit
    while iteration < max_iterations:
        job_index = np.random.randint(0, len(jobs) - 1)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(all_shifts)
        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            new_objective, new_overload, delta_tardiness = evaluate_move(
                load_at_time, job, new_start, old_start, c_t, total_tardiness
            )

            # Simulated annealing acceptance criteria
            if new_objective <= current_objective or np.random.random() < np.exp((current_objective - new_objective) / current_temp):
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] = max(0, new_start + job['processing_time'] - job['due_date'])
                total_tardiness = np.sum(tardiness_values)
                total_overload = new_overload
                current_objective = new_objective
                if new_objective < best_objective:
                    best_objective = new_objective

        # Cooling schedule for simulated annealing
        current_temp *= cooling_rate
        current_temp = max(current_temp, 1)  # Ensure temperature does not fall below 1

        iteration += 1  # Increment iteration count

    return best_objective

def get_cooling_rate(num_jobs):
    """
    Returns the cooling rate based on the number of jobs.
    """
    if num_jobs <= 150:
        return 0.9999
    elif 151 <= num_jobs <= 300:
        return 0.99995
    elif 301 <= num_jobs <= 600:
        return 0.99999
    else:
        return 0.999995  # For jobs > 600

def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 15, 16, 3, 0, 10, 20), (2, 19, 16, 117, 91, 10, 20), (3, 6, 5, 15, 9, 10, 20), (4, 17, 15, 121, 95, 10, 20), (5, 19, 13, 121, 94, 10, 20), (6, 16, 14, 25, 28, 10, 20), (7, 5, 7, 47, 62, 10, 20), (8, 18, 14, 7, 9, 10, 20), (9, 17, 14, 31, 18, 10, 20), (10, 15, 16, 76, 62, 10, 20), (11, 20, 11, 7, 4, 10, 20), (12, 17, 17, 29, 2, 10, 20), (13, 10, 9, 117, 125, 10, 20), (14, 15, 20, 121, 102, 10, 20), (15, 9, 14, 90, 88, 10, 20), (16, 5, 8, 104, 116, 10, 20), (17, 11, 17, 50, 33, 10, 20), (18, 17, 6, 39, 42, 10, 20), (19, 20, 13, 7, 2, 10, 20), (20, 9, 5, 55, 36, 10, 20), (21, 12, 10, 55, 60, 10, 20), (22, 7, 7, 62, 70, 10, 20), (23, 17, 20, 47, 33, 10, 20), (24, 10, 20, 34, 26, 10, 20), (25, 5, 18, 48, 52, 10, 20), (26, 13, 5, 34, 33, 10, 20), (27, 7, 17, 28, 14, 10, 20), (28, 6, 9, 49, 57, 10, 20), (29, 15, 12, 76, 75, 10, 20), (30, 9, 16, 48, 46, 10, 20), (31, 14, 20, 117, 122, 10, 20), (32, 12, 15, 28, 16, 10, 20), (33, 6, 17, 121, 131, 10, 20), (34, 9, 13, 15, 23, 10, 20), (35, 10, 19, 48, 57, 10, 20), (36, 12, 7, 15, 17, 10, 20), (37, 10, 20, 90, 96, 10, 20), (38, 11, 6, 121, 108, 10, 20), (39, 15, 6, 49, 31, 10, 20), (40, 17, 16, 55, 42, 10, 20), (41, 16, 7, 34, 18, 10, 20), (42, 7, 16, 31, 23, 10, 20), (43, 11, 5, 121, 103, 10, 20), (44, 14, 13, 124, 108, 10, 20), (45, 20, 9, 90, 84, 10, 20), (46, 14, 10, 15, 20, 10, 20), (47, 5, 7, 55, 61, 10, 20), (48, 19, 18, 15, 10, 10, 20), (49, 15, 11, 28, 33, 10, 20), (50, 8, 14, 3, 13, 10, 20), (51, 20, 15, 34, 29, 10, 20), (52, 11, 9, 49, 43, 10, 20), (53, 6, 10, 29, 13, 10, 20), (54, 12, 19, 34, 32, 10, 20), (55, 7, 13, 121, 134, 10, 20), (56, 18, 15, 49, 48, 10, 20), (57, 16, 20, 26, 19, 10, 20), (58, 13, 15, 124, 130, 10, 20), (59, 10, 14, 76, 74, 10, 20), (60, 12, 11, 121, 100, 10, 20), (61, 5, 9, 26, 37, 10, 20), (62, 11, 14, 76, 56, 10, 20), (63, 20, 7, 3, 0, 10, 20), (64, 19, 5, 31, 25, 10, 20), (65, 6, 13, 76, 87, 10, 20), (66, 8, 5, 49, 34, 10, 20), (67, 8, 6, 76, 58, 10, 20), (68, 5, 20, 55, 51, 10, 20), (69, 10, 18, 62, 63, 10, 20), (70, 19, 16, 15, 1, 10, 20), (71, 14, 19, 47, 46, 10, 20), (72, 14, 10, 34, 11, 10, 20), (73, 8, 9, 90, 81, 10, 20), (74, 20, 8, 117, 101, 10, 20), (75, 20, 8, 26, 26, 10, 20), (76, 13, 6, 28, 21, 10, 20), (77, 15, 8, 55, 60, 10, 20), (78, 11, 11, 15, 22, 10, 20), (79, 17, 10, 121, 111, 10, 20), (80, 7, 15, 124, 135, 10, 20), (81, 10, 12, 104, 104, 10, 20), (82, 6, 12, 130, 139, 10, 20), (83, 18, 8, 130, 129, 10, 20), (84, 10, 18, 7, 1, 10, 20), (85, 13, 9, 90, 77, 10, 20), (86, 8, 19, 25, 7, 10, 20), (87, 7, 11, 29, 14, 10, 20), (88, 19, 11, 28, 27, 10, 20), (89, 8, 10, 26, 33, 10, 20), (90, 13, 18, 48, 27, 10, 20), (91, 16, 10, 39, 21, 10, 20), (92, 18, 9, 55, 42, 10, 20), (93, 18, 16, 62, 63, 10, 20), (94, 14, 13, 90, 86, 10, 20), (95, 11, 6, 15, 1, 10, 20), (96, 11, 18, 121, 123, 10, 20), (97, 10, 10, 55, 46, 10, 20), (98, 16, 10, 49, 30, 10, 20), (99, 5, 10, 130, 139, 10, 20), (100, 11, 20, 25, 31, 10, 20)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    c_t = 95
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    iteration_values = [25000, 40000, 70000, 90000, 120000]
    max_plateau_moves = 8000
    
    
    cooling_rate = 0.99995
    initial_temp = 10*len(jobs)

    
    for max_iterations in iteration_values:

        objective_values = []
        durations_list = []

        for i in range(32):
            seed_value = seeds[i]

            start_time = time.time()
            # Run hybrid LS+SA using the estimated initial temperature and current cooling rate
            best_objective = hybrid_local_search_simulated_annealing(
                jobs, c_t, max_time, max_iterations, max_plateau_moves, seed_value, initial_temp, cooling_rate
            )
            duration = time.time() - start_time

            if i > 1:  # Exclude the first two runs for JIT compilation
                objective_values.append(best_objective)
                durations_list.append(duration)

        # Calculate summary statistics for the last 30 runs
        if objective_values:  # Ensure there are valid results
            mean_objective = np.mean(objective_values)
            std_dev_objective = np.std(objective_values)
            min_objective = np.min(objective_values)
            max_objective = np.max(objective_values)
            q1_objective = np.percentile(objective_values, 25)
            q3_objective = np.percentile(objective_values, 75)
            average_duration = np.mean(durations_list)

            # Print summary for the current configuration
            print("\nSummary for Max Iterations = ", max_iterations)
            print(f"Mean Objective Value: {mean_objective:.2f}")
            print(f"Standard Deviation: {std_dev_objective:.2f}")
            print(f"Minimum Objective Value: {min_objective:.2f}")
            print(f"Maximum Objective Value: {max_objective:.2f}")
            print(f"Q1 Objective (25th percentile): {q1_objective:.2f}")
            print(f"Q3 Objective (75th percentile): {q3_objective:.2f}")
            print(f"Average Duration: {average_duration:.3f} seconds")


if __name__ == '__main__':
    main()