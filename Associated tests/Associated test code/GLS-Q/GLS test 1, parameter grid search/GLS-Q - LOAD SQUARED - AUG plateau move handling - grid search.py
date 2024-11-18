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
def calculate_utilities(load_at_quarter, penalties):
    # Calculate the utilities for each quarter
    utilities = np.zeros(4, dtype=np.float64)
    for q in range(4):
        utilities[q] = load_at_quarter[q] / (1.0 + penalties[q])
    return utilities

@njit(cache=True)
def calculate_objective_value(total_overload, total_tardiness):
    return 10 * total_overload + total_tardiness

@njit(cache=True)
def calculate_augmented_objective(original_objective, load_at_time, penalties, nu, max_time):
    augmented_objective = original_objective
    quarter_size = max_time // 4
    
    # Loop through each quarter
    for q in range(4):
        start = q * quarter_size
        if q == 3:  # Last quarter includes any residual time periods
            end = max_time + 1
        else:
            end = (q + 1) * quarter_size
        
        # Penalize individual time periods within the quarter
        for t in range(start, end):
            augmented_objective += nu * penalties[q] * (load_at_time[t] ** 2)
    
    return augmented_objective


@njit(cache=True)
def is_within_constraints(job, new_start):
    job_end = new_start + job['processing_time']
    lower_bound = job['due_date'] - job['lower_bound']
    upper_bound = job['due_date'] + job['upper_bound']
    return lower_bound <= job_end <= upper_bound

@njit(cache=True)
def calculate_quarterly_load(load_at_time, max_time):
    # Calculate the number of time periods in each quarter
    quarter_size = max_time // 4
    quarters = np.zeros(4, dtype=np.int32)

    # Sum the load for each quarter
    for i in range(4):
        start = i * quarter_size
        if i == 3:  # Last quarter includes any residual time periods
            end = max_time + 1
        else:
            end = (i + 1) * quarter_size
        quarters[i] = np.sum(load_at_time[start:end])
    
    return quarters

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

    # Calculate the augmented objective
    augmented_objective = calculate_augmented_objective(original_objective, load_at_time, penalties, nu, max_time)

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
    all_shifts = np.arange(-20, 21)
    plateau_moves = 0  # Initialize counter for plateau moves
    penalties = np.zeros(4, dtype=np.int32)  # Initialize penalty array
    best_original_objective = current_objective  # Track the best original objective value
    best_augmented_objective = float('inf')  # Initialize the best augmented objective value

    iteration = 0  # Initialize iteration counter

    while iteration < max_iterations:
        # If plateau moves exceed the limit, apply the penalty mechanism
        if plateau_moves >= max_plateau_moves:
            # Calculate the load at each quarter
            quarterly_load = calculate_quarterly_load(load_at_time, max_time)
            # Calculate utilities for each quarter based on the current penalties
            utilities = calculate_utilities(quarterly_load, penalties)
            # Find the quarter with the highest utility
            quarter_to_penalize = np.argmax(utilities)
            # Increase the penalty for the quarter with the highest utility
            penalties[quarter_to_penalize] += 1

            # Reset penalties if their sum equals or exceeds the current threshold
            if np.sum(penalties) >= penalty_reset_threshold:
                penalties[:] = 0  # Reset all penalties to zero

            # Reset plateau moves
            plateau_moves = 0

            # Force a move after penalizing
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

                # Stick current objective to the new augmented objective
                current_objective = augmented_objective

            iteration += 1
            continue  # Skip to the next iteration after forcing the move

        # Pick a random job and a random shift
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

    # Calculate the quarterly load at the end of the search
    quarterly_load = calculate_quarterly_load(load_at_time, max_time)
    
    # Calculate the final augmented objective value using the current penalties and quarterly loads
    final_augmented_objective = calculate_augmented_objective(current_objective, load_at_time, penalties, nu, max_time)
    
    # Return best original objective, final augmented objective, quarterly load, penalties, and final current objective
    return best_original_objective, final_augmented_objective


def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 18, 16, 196, 182, 20, 40), (2, 29, 32, 0, 0, 20, 40), (3, 32, 31, 205, 164, 20, 40), (4, 23, 12, 72, 29, 20, 40), (5, 30, 38, 111, 100, 20, 40), (6, 35, 35, 162, 143, 20, 40), (7, 37, 24, 257, 213, 20, 40), (8, 16, 25, 125, 99, 20, 40), (9, 39, 29, 125, 117, 20, 40), (10, 39, 37, 162, 134, 20, 40), (11, 36, 16, 243, 219, 20, 40), (12, 12, 25, 93, 75, 20, 40), (13, 21, 10, 224, 235, 20, 40), (14, 36, 25, 205, 151, 20, 40), (15, 39, 28, 60, 39, 20, 40), (16, 23, 19, 224, 223, 20, 40), (17, 36, 29, 57, 17, 20, 40), (18, 21, 13, 109, 101, 20, 40), (19, 19, 17, 185, 178, 20, 40), (20, 31, 33, 145, 152, 20, 40), (21, 34, 15, 162, 121, 20, 40), (22, 14, 27, 36, 14, 20, 40), (23, 31, 38, 94, 45, 20, 40), (24, 11, 13, 157, 182, 20, 40), (25, 10, 27, 60, 85, 20, 40), (26, 28, 19, 192, 198, 20, 40), (27, 19, 19, 166, 174, 20, 40), (28, 30, 30, 94, 61, 20, 40), (29, 31, 18, 111, 83, 20, 40), (30, 35, 37, 109, 98, 20, 40), (31, 20, 36, 109, 120, 20, 40), (32, 29, 34, 5, 16, 20, 40), (33, 19, 30, 6, 12, 20, 40), (34, 38, 18, 88, 63, 20, 40), (35, 26, 39, 188, 155, 20, 40), (36, 14, 25, 186, 160, 20, 40), (37, 37, 33, 185, 185, 20, 40), (38, 35, 13, 185, 174, 20, 40), (39, 13, 16, 43, 70, 20, 40), (40, 23, 35, 125, 131, 20, 40), (41, 29, 14, 5, 10, 20, 40), (42, 39, 40, 125, 110, 20, 40), (43, 21, 36, 85, 84, 20, 40), (44, 30, 31, 3, 6, 20, 40), (45, 28, 10, 43, 47, 20, 40), (46, 16, 27, 147, 149, 20, 40), (47, 31, 14, 38, 10, 20, 40), (48, 34, 33, 224, 210, 20, 40), (49, 25, 32, 38, 36, 20, 40), (50, 15, 32, 5, 14, 20, 40), (51, 19, 10, 0, 18, 20, 40), (52, 28, 35, 166, 146, 20, 40), (53, 32, 32, 192, 160, 20, 40), (54, 36, 27, 185, 131, 20, 40), (55, 27, 34, 195, 153, 20, 40), (56, 14, 22, 177, 199, 20, 40), (57, 29, 13, 93, 77, 20, 40), (58, 21, 15, 177, 180, 20, 40), (59, 22, 14, 257, 254, 20, 40), (60, 25, 35, 109, 75, 20, 40), (61, 39, 28, 232, 189, 20, 40), (62, 39, 31, 147, 96, 20, 40), (63, 25, 34, 157, 121, 20, 40), (64, 28, 27, 166, 131, 20, 40), (65, 30, 19, 177, 142, 20, 40), (66, 21, 19, 3, 10, 20, 40), (67, 32, 15, 205, 190, 20, 40), (68, 29, 15, 43, 10, 20, 40), (69, 39, 35, 60, 23, 20, 40), (70, 20, 28, 222, 228, 20, 40), (71, 27, 13, 46, 37, 20, 40), (72, 38, 40, 147, 94, 20, 40), (73, 27, 25, 60, 22, 20, 40), (74, 29, 17, 196, 191, 20, 40), (75, 18, 34, 224, 217, 20, 40), (76, 33, 27, 189, 177, 20, 40), (77, 16, 33, 177, 157, 20, 40), (78, 31, 21, 166, 149, 20, 40), (79, 20, 11, 60, 60, 20, 40), (80, 30, 11, 255, 227, 20, 40), (81, 30, 18, 85, 89, 20, 40), (82, 16, 11, 222, 196, 20, 40), (83, 22, 26, 145, 105, 20, 40), (84, 10, 21, 111, 133, 20, 40), (85, 38, 10, 232, 188, 20, 40), (86, 13, 26, 177, 163, 20, 40), (87, 20, 23, 162, 171, 20, 40), (88, 37, 29, 196, 155, 20, 40), (89, 13, 29, 224, 247, 20, 40), (90, 16, 36, 195, 178, 20, 40), (91, 30, 29, 192, 196, 20, 40), (92, 40, 37, 189, 173, 20, 40), (93, 23, 38, 157, 167, 20, 40), (94, 25, 32, 57, 55, 20, 40), (95, 26, 31, 5, 9, 20, 40), (96, 36, 27, 162, 127, 20, 40), (97, 24, 22, 157, 157, 20, 40), (98, 20, 22, 120, 100, 20, 40), (99, 11, 19, 36, 52, 20, 40), (100, 34, 16, 94, 100, 20, 40), (101, 35, 17, 22, 17, 20, 40), (102, 40, 36, 72, 51, 20, 40), (103, 18, 38, 109, 124, 20, 40), (104, 30, 31, 43, 53, 20, 40), (105, 24, 34, 38, 3, 20, 40), (106, 33, 22, 260, 238, 20, 40), (107, 37, 35, 145, 137, 20, 40), (108, 14, 21, 60, 28, 20, 40), (109, 16, 12, 196, 161, 20, 40), (110, 23, 24, 196, 201, 20, 40), (111, 39, 33, 147, 107, 20, 40), (112, 28, 29, 88, 70, 20, 40), (113, 31, 17, 94, 47, 20, 40), (114, 39, 39, 3, 2, 20, 40), (115, 10, 16, 5, 0, 20, 40), (116, 39, 33, 5, 1, 20, 40), (117, 22, 23, 145, 136, 20, 40), (118, 35, 30, 186, 150, 20, 40), (119, 33, 13, 185, 166, 20, 40), (120, 26, 17, 260, 259, 20, 40), (121, 29, 23, 189, 165, 20, 40), (122, 36, 19, 3, 6, 20, 40), (123, 18, 16, 22, 21, 20, 40), (124, 18, 37, 88, 82, 20, 40), (125, 20, 23, 224, 206, 20, 40), (126, 40, 26, 46, 5, 20, 40), (127, 33, 22, 36, 43, 20, 40), (128, 15, 25, 162, 141, 20, 40), (129, 36, 30, 117, 65, 20, 40), (130, 28, 23, 222, 230, 20, 40), (131, 32, 24, 162, 126, 20, 40), (132, 12, 26, 72, 54, 20, 40), (133, 40, 11, 125, 113, 20, 40), (134, 37, 28, 195, 151, 20, 40), (135, 16, 31, 192, 191, 20, 40), (136, 37, 39, 145, 123, 20, 40), (137, 38, 21, 260, 219, 20, 40), (138, 34, 22, 88, 91, 20, 40), (139, 26, 24, 257, 233, 20, 40), (140, 19, 36, 184, 195, 20, 40), (141, 35, 22, 120, 111, 20, 40), (142, 31, 23, 22, 4, 20, 40), (143, 37, 23, 205, 161, 20, 40), (144, 22, 28, 224, 197, 20, 40), (145, 19, 20, 257, 232, 20, 40), (146, 11, 23, 192, 185, 20, 40), (147, 37, 26, 140, 133, 20, 40), (148, 20, 14, 109, 70, 20, 40), (149, 11, 26, 260, 278, 20, 40), (150, 40, 25, 189, 165, 20, 40), (151, 32, 16, 125, 95, 20, 40), (152, 25, 15, 257, 220, 20, 40), (153, 13, 18, 85, 87, 20, 40), (154, 13, 23, 72, 97, 20, 40), (155, 21, 31, 120, 116, 20, 40), (156, 29, 14, 43, 21, 20, 40), (157, 26, 36, 43, 11, 20, 40), (158, 36, 29, 257, 250, 20, 40), (159, 36, 28, 257, 250, 20, 40), (160, 40, 31, 196, 186, 20, 40), (161, 38, 18, 224, 198, 20, 40), (162, 23, 10, 22, 32, 20, 40), (163, 26, 39, 94, 57, 20, 40), (164, 27, 27, 120, 97, 20, 40), (165, 18, 26, 140, 113, 20, 40), (166, 17, 11, 224, 190, 20, 40), (167, 31, 24, 60, 54, 20, 40), (168, 21, 20, 72, 47, 20, 40), (169, 18, 16, 195, 166, 20, 40), (170, 10, 10, 85, 67, 20, 40), (171, 33, 39, 195, 182, 20, 40), (172, 28, 14, 232, 192, 20, 40), (173, 19, 37, 6, 22, 20, 40), (174, 38, 32, 243, 209, 20, 40), (175, 39, 17, 46, 3, 20, 40), (176, 37, 31, 109, 58, 20, 40), (177, 14, 19, 125, 107, 20, 40), (178, 14, 30, 188, 174, 20, 40), (179, 39, 21, 162, 144, 20, 40), (180, 12, 33, 5, 4, 20, 40), (181, 40, 17, 192, 138, 20, 40), (182, 22, 35, 157, 121, 20, 40), (183, 39, 39, 109, 85, 20, 40), (184, 39, 28, 166, 131, 20, 40), (185, 13, 37, 125, 114, 20, 40), (186, 13, 17, 196, 213, 20, 40), (187, 36, 25, 222, 186, 20, 40), (188, 21, 36, 46, 53, 20, 40), (189, 15, 23, 120, 105, 20, 40), (190, 17, 28, 125, 147, 20, 40), (191, 36, 11, 3, 7, 20, 40), (192, 40, 11, 205, 171, 20, 40), (193, 27, 22, 157, 150, 20, 40), (194, 13, 39, 36, 36, 20, 40), (195, 15, 17, 38, 63, 20, 40), (196, 21, 33, 255, 233, 20, 40), (197, 22, 40, 257, 238, 20, 40), (198, 19, 14, 88, 107, 20, 40), (199, 36, 33, 185, 134, 20, 40), (200, 26, 17, 196, 159, 20, 40)

], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    c_t = 610
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_iterations = 1250000
    max_stagnant_moves = 32000
    nu = 1  # Define the penalty multiplier nu
    
    results = []
    
    # Iterate over penalty reset thresholds from 2 to 20
    for penalty_reset_threshold in range(2, 21):
        best_original_objective, augmented_objective = simplified_random_local_search(
            jobs, c_t, max_time, max_iterations, max_stagnant_moves, seeds[0], nu, penalty_reset_threshold
        )
        results.append((penalty_reset_threshold, best_original_objective, augmented_objective))
    
    # Sort results by best_original_objective
    results.sort(key=lambda x: x[1])
    
    # Get the top 5 runs with the lowest original objective value
    top_5_runs = results[:5]
    
    # Print the top 5 runs
    print("Top 5 runs based on original objective value:")
    for run in top_5_runs:
        print(f"Penalty Reset Threshold: {run[0]}, Best Original Objective: {run[1]}, Augmented Objective: {run[2]}")

if __name__ == '__main__':
    main()