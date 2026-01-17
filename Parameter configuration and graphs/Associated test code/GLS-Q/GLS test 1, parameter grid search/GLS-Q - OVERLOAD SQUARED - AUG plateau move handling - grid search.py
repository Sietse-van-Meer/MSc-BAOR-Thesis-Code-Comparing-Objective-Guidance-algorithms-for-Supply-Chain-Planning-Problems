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
def calculate_utilities(quarterly_overload, penalties):
    # Calculate the utilities for each quarter based on overload
    utilities = np.zeros(4, dtype=np.float64)
    for q in range(4):
        utilities[q] = quarterly_overload[q] / (1.0 + penalties[q])
    return utilities

@njit(cache=True)
def calculate_objective_value(total_overload, total_tardiness):
    return 10 * total_overload + total_tardiness

@njit(cache=True)
def calculate_augmented_objective(original_objective, quarterly_overload, penalties, nu):
    augmented_objective = original_objective
    # Apply penalty by squaring individual overloads within each quarter, then summing
    for q in range(4):
        augmented_objective += nu * penalties[q] * quarterly_overload[q]
    return augmented_objective

@njit(cache=True)
def is_within_constraints(job, new_start):
    job_end = new_start + job['processing_time']
    lower_bound = job['due_date'] - job['lower_bound']
    upper_bound = job['due_date'] + job['upper_bound']
    return lower_bound <= job_end <= upper_bound

@njit(cache=True)
def calculate_quarterly_overload(load_at_time, c_t, max_time):
    # Calculate the number of time periods in each quarter
    quarter_size = max_time // 4
    quarterly_overload = np.zeros(4, dtype=np.float64)

    # Sum of squared overloads for each quarter
    for i in range(4):
        start = i * quarter_size
        if i == 3:  # Last quarter includes any residual time periods
            end = max_time + 1
        else:
            end = (i + 1) * quarter_size

        # Calculate overload for this quarter (square of max(load - c_t, 0) per time period)
        overloads = np.maximum(0, load_at_time[start:end] - c_t)
        quarterly_overload[i] = np.sum(overloads**2)  # Square individual overloads and sum
    return quarterly_overload

@njit(cache=True)
def evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness, penalties, nu, max_time):
    duration = job['processing_time']
    weight = job['weight']

    # Temporarily adjust load for the new start
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] += weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] -= weight

    # Calculate the new overload and original objective
    new_overload = calculate_total_overload(load_at_time, c_t)
    new_tardiness = max(0, new_start + duration - job['due_date'])
    delta_tardiness = new_tardiness - max(0, old_start + duration - job['due_date'])
    original_objective = calculate_objective_value(new_overload, total_tardiness + delta_tardiness)

    # Calculate the quarterly overload for the new state
    quarterly_overload = calculate_quarterly_overload(load_at_time, c_t, max_time)

    # Calculate the augmented objective using the squared overloads per time period
    augmented_objective = calculate_augmented_objective(original_objective, quarterly_overload, penalties, nu)

    # Revert the temporary adjustment
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] -= weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] += weight

    return augmented_objective, original_objective, new_overload, delta_tardiness

@njit(cache=True)
def simplified_random_local_search(jobs, c_t, max_time, max_iterations, max_plateau_moves, seed_value, nu, penalty_reset_threshold):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    load_at_time = calculate_load_at_time(current_solution, max_time)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    total_tardiness = np.sum(tardiness_values)
    total_overload = calculate_total_overload(load_at_time, c_t)
    current_objective = calculate_objective_value(total_overload, total_tardiness)

    job_indices = np.arange(len(jobs))
    all_shifts = np.arange(-10, 11)
    plateau_moves = 0
    penalties = np.zeros(4, dtype=np.int32)
    best_original_objective = current_objective

    iteration = 0

    while iteration < max_iterations:
        if plateau_moves >= max_plateau_moves:
            quarterly_overload = calculate_quarterly_overload(load_at_time, c_t, max_time)
            utilities = calculate_utilities(quarterly_overload, penalties)
            quarter_to_penalize = np.argmax(utilities)
            penalties[quarter_to_penalize] += 1
            if np.sum(penalties) >= penalty_reset_threshold:
                penalties[:] = 0
            plateau_moves = 0

            job_index = np.random.choice(job_indices)
            job = current_solution[job_index]
            old_start = job['start_time']
            shift = np.random.choice(all_shifts)
            new_start = old_start + shift

            if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
                augmented_objective, original_objective, new_overload, delta_tardiness = evaluate_move(
                    load_at_time, job, new_start, old_start, c_t, total_tardiness, penalties, nu, max_time
                )
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] += delta_tardiness
                total_tardiness += delta_tardiness
                total_overload = new_overload
                current_objective = augmented_objective

            iteration += 1
            continue

        job_index = np.random.choice(job_indices)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(all_shifts)

        if shift == 0:
            continue

        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            augmented_objective, original_objective, new_overload, delta_tardiness = evaluate_move(
                load_at_time, job, new_start, old_start, c_t, total_tardiness, penalties, nu, max_time
            )
            iteration += 1  # Increment iteration count

            if augmented_objective <= current_objective:  # Accept the move if augmented objective improves or stays the same
                if original_objective < best_original_objective:  # Update only if original objective improves
                    best_original_objective = original_objective

                if augmented_objective < current_objective:  # Reset plateau moves if augmented objective improves
                    plateau_moves = 0
                    best_augmented_objective = augmented_objective  # Track the best augmented objective value
                else:
                    plateau_moves += 1  # Increment plateau moves if the augmented objective stays the same

                # Make the move
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] += delta_tardiness
                total_tardiness += delta_tardiness
                total_overload = new_overload
                current_objective = augmented_objective  # Update to augmented objective
            else:
                plateau_moves += 1  # Increment plateau moves if no improvement

    quarterly_overload = calculate_quarterly_overload(load_at_time, c_t, max_time)
    final_augmented_objective = calculate_augmented_objective(current_objective, quarterly_overload, penalties, nu)

    return best_original_objective, final_augmented_objective


def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 9, 13, 206, 195, 10, 20), (2, 19, 10, 82, 79, 10, 20), (3, 10, 20, 82, 71, 10, 20), (4, 15, 20, 156, 159, 10, 20), (5, 10, 15, 167, 160, 10, 20), (6, 8, 6, 80, 65, 10, 20), (7, 11, 13, 57, 43, 10, 20), (8, 17, 16, 175, 159, 10, 20), (9, 15, 7, 104, 107, 10, 20), (10, 14, 9, 156, 140, 10, 20), (11, 8, 12, 70, 57, 10, 20), (12, 5, 16, 190, 203, 10, 20), (13, 10, 10, 190, 186, 10, 20), (14, 15, 5, 167, 152, 10, 20), (15, 19, 7, 229, 218, 10, 20), (16, 18, 16, 65, 67, 10, 20), (17, 8, 9, 112, 114, 10, 20), (18, 11, 16, 136, 123, 10, 20), (19, 9, 8, 181, 187, 10, 20), (20, 9, 6, 57, 62, 10, 20), (21, 13, 9, 79, 76, 10, 20), (22, 5, 12, 56, 49, 10, 20), (23, 13, 9, 80, 81, 10, 20), (24, 11, 16, 175, 155, 10, 20), (25, 17, 19, 80, 63, 10, 20), (26, 20, 18, 201, 196, 10, 20), (27, 5, 8, 152, 161, 10, 20), (28, 11, 16, 175, 159, 10, 20), (29, 12, 8, 161, 168, 10, 20), (30, 9, 7, 65, 65, 10, 20), (31, 19, 16, 57, 58, 10, 20), (32, 9, 17, 224, 233, 10, 20), (33, 10, 16, 224, 232, 10, 20), (34, 10, 6, 136, 121, 10, 20), (35, 5, 13, 80, 67, 10, 20), (36, 16, 19, 181, 181, 10, 20), (37, 15, 13, 156, 148, 10, 20), (38, 11, 19, 65, 51, 10, 20), (39, 15, 19, 201, 202, 10, 20), (40, 12, 8, 229, 212, 10, 20), (41, 8, 19, 161, 158, 10, 20), (42, 9, 10, 190, 192, 10, 20), (43, 13, 15, 161, 165, 10, 20), (44, 9, 9, 190, 185, 10, 20), (45, 16, 18, 224, 205, 10, 20), (46, 11, 7, 230, 234, 10, 20), (47, 5, 20, 136, 139, 10, 20), (48, 14, 17, 224, 215, 10, 20), (49, 18, 14, 190, 173, 10, 20), (50, 14, 15, 70, 56, 10, 20), (51, 11, 5, 156, 162, 10, 20), (52, 13, 16, 104, 96, 10, 20), (53, 17, 6, 181, 168, 10, 20), (54, 20, 8, 175, 165, 10, 20), (55, 13, 8, 82, 81, 10, 20), (56, 10, 9, 156, 150, 10, 20), (57, 16, 15, 152, 130, 10, 20), (58, 20, 13, 206, 179, 10, 20), (59, 9, 8, 112, 114, 10, 20), (60, 11, 19, 201, 189, 10, 20), (61, 14, 18, 175, 164, 10, 20), (62, 6, 9, 80, 65, 10, 20), (63, 10, 7, 161, 163, 10, 20), (64, 19, 13, 136, 116, 10, 20), (65, 20, 14, 206, 203, 10, 20), (66, 16, 14, 57, 38, 10, 20), (67, 9, 15, 224, 225, 10, 20), (68, 20, 13, 175, 167, 10, 20), (69, 14, 7, 3, 2, 10, 20), (70, 16, 10, 82, 57, 10, 20), (71, 10, 13, 152, 145, 10, 20), (72, 17, 20, 181, 163, 10, 20), (73, 14, 8, 175, 162, 10, 20), (74, 7, 8, 167, 175, 10, 20), (75, 12, 11, 104, 90, 10, 20), (76, 9, 18, 175, 168, 10, 20), (77, 14, 6, 230, 209, 10, 20), (78, 10, 5, 181, 165, 10, 20), (79, 11, 20, 65, 62, 10, 20), (80, 10, 14, 152, 135, 10, 20), (81, 10, 12, 229, 227, 10, 20), (82, 14, 9, 201, 193, 10, 20), (83, 18, 8, 201, 192, 10, 20), (84, 17, 17, 181, 159, 10, 20), (85, 10, 8, 229, 238, 10, 20), (86, 7, 19, 57, 41, 10, 20), (87, 11, 16, 112, 114, 10, 20), (88, 10, 16, 136, 127, 10, 20), (89, 19, 6, 79, 63, 10, 20), (90, 6, 11, 161, 170, 10, 20), (91, 8, 15, 167, 161, 10, 20), (92, 19, 12, 56, 47, 10, 20), (93, 8, 12, 70, 73, 10, 20), (94, 17, 9, 112, 112, 10, 20), (95, 16, 17, 161, 165, 10, 20), (96, 17, 9, 224, 210, 10, 20), (97, 5, 7, 224, 224, 10, 20), (98, 9, 16, 181, 179, 10, 20), (99, 17, 9, 57, 38, 10, 20), (100, 12, 20, 57, 45, 10, 20)

], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    c_t = 80
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_iterations = 176000
    max_stagnant_moves = 8000
    nu = 1  # Define the penalty multiplier nu
    
    results = []
    
    for penalty_reset_threshold in range(2, 21):
        best_original_objective, augmented_objective = simplified_random_local_search(
            jobs, c_t, max_time, max_iterations, max_stagnant_moves, seeds[0], nu, penalty_reset_threshold
        )
        results.append((penalty_reset_threshold, best_original_objective, augmented_objective))

    results.sort(key=lambda x: x[1])
    
    top_5_runs = results[:5]

    print("Top 5 runs based on original objective value:")
    for run in top_5_runs:
        print(f"Penalty Reset Threshold: {run[0]}, Best Original Objective: {run[1]}")

if __name__ == '__main__':
    main()