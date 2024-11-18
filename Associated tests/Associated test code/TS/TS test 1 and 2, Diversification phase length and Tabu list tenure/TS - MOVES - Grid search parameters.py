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
def simplified_random_local_search(jobs, c_t, max_time, seed_value, diversification_moves, tenure, max_iterations):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    max_plateau_moves = 32000
    tabu_list = np.zeros((tenure, 2), dtype=np.int32)  # Tabu list now stores tuples of (job_index, shift)

    load_at_time = calculate_load_at_time(current_solution, max_time)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    total_tardiness = np.sum(tardiness_values)
    total_overload = calculate_total_overload(load_at_time, c_t)
    current_objective = calculate_objective_value(total_overload, total_tardiness)
    best_objective = current_objective  # Initialize the best objective found

    job_indices = np.arange(len(jobs))
    all_shifts = np.arange(-20, 21)
    all_shifts = all_shifts[all_shifts != 0]  # Remove zero shift (no move)
    plateau_moves = 0  # Initialize counter for plateau moves

    iteration = 0
    while iteration < max_iterations:
        if plateau_moves >= max_plateau_moves:
            tabu_list[:, :] = 0  # Clear tabu list during diversification
            for _ in range(diversification_moves):  # Perform exactly diversification_moves moves
                job_index = np.random.choice(job_indices)
                job = current_solution[job_index]
                old_start = job['start_time']
                shift = np.random.choice(all_shifts)
                new_start = old_start + shift

                if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] = max(0, new_start + job['processing_time'] - job['due_date'])
                    total_tardiness = np.sum(tardiness_values)
                    total_overload = calculate_total_overload(load_at_time, c_t)
                    current_objective = calculate_objective_value(total_overload, total_tardiness)
                    if current_objective < best_objective:
                        best_objective = current_objective  # Update the best objective if necessary
            plateau_moves = 0  # Reset plateau moves after diversification

        # Normal operation outside of diversification
        job_index = np.random.choice(job_indices)
        shift = np.random.choice(all_shifts)

        # Check if the (job_index, shift) tuple is in the tabu list
        is_tabu = False
        for i in range(tenure):
            if tabu_list[i, 0] == job_index and tabu_list[i, 1] == shift:
                is_tabu = True
                break

        if is_tabu:
            continue

        job = current_solution[job_index]
        old_start = job['start_time']
        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            new_objective, new_overload, delta_tardiness = evaluate_move(
                load_at_time, job, new_start, old_start, c_t, total_tardiness
            )

            if new_objective <= current_objective:
                if new_objective < current_objective:
                    plateau_moves = 0
                    if new_objective < best_objective:
                        best_objective = new_objective  # Update the best objective found
                else:
                    plateau_moves += 1

                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] += delta_tardiness
                total_tardiness += delta_tardiness
                total_overload = new_overload
                current_objective = new_objective

                # Manual roll: shift all entries in the tabu list one step to the left
                for i in range(tenure - 1):
                    tabu_list[i] = tabu_list[i + 1]
                tabu_list[-1] = (job_index, shift)  # Add new job-move pair to tabu

            else:
                plateau_moves += 1

        iteration += 1

    return best_objective  # Return the best objective found


def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 36, 32, 162, 120, 20, 40), (2, 40, 23, 357, 312, 20, 40), (3, 28, 19, 400, 401, 20, 40), (4, 10, 27, 429, 416, 20, 40), (5, 12, 35, 50, 65, 20, 40), (6, 38, 11, 440, 418, 20, 40), (7, 39, 35, 47, 14, 20, 40), (8, 18, 16, 295, 301, 20, 40), (9, 20, 17, 25, 11, 20, 40), (10, 37, 35, 117, 112, 20, 40), (11, 30, 27, 50, 38, 20, 40), (12, 23, 24, 302, 293, 20, 40), (13, 14, 13, 400, 382, 20, 40), (14, 13, 18, 193, 200, 20, 40), (15, 33, 37, 81, 38, 20, 40), (16, 36, 40, 429, 380, 20, 40), (17, 36, 32, 269, 246, 20, 40), (18, 25, 20, 402, 399, 20, 40), (19, 26, 14, 167, 167, 20, 40), (20, 11, 31, 283, 278, 20, 40), (21, 22, 19, 448, 432, 20, 40), (22, 20, 14, 221, 223, 20, 40), (23, 26, 36, 395, 398, 20, 40), (24, 34, 38, 297, 275, 20, 40), (25, 26, 26, 25, 8, 20, 40), (26, 13, 34, 130, 152, 20, 40), (27, 20, 18, 167, 165, 20, 40), (28, 14, 14, 377, 348, 20, 40), (29, 16, 32, 387, 356, 20, 40), (30, 13, 12, 373, 342, 20, 40), (31, 37, 38, 205, 208, 20, 40), (32, 32, 27, 162, 151, 20, 40), (33, 20, 24, 205, 172, 20, 40), (34, 32, 13, 117, 81, 20, 40), (35, 23, 29, 191, 155, 20, 40), (36, 28, 28, 402, 355, 20, 40), (37, 29, 15, 295, 266, 20, 40), (38, 23, 26, 440, 429, 20, 40), (39, 30, 19, 357, 350, 20, 40), (40, 26, 33, 387, 372, 20, 40), (41, 23, 35, 429, 422, 20, 40), (42, 24, 30, 264, 259, 20, 40), (43, 29, 10, 283, 254, 20, 40), (44, 25, 13, 69, 28, 20, 40), (45, 36, 23, 269, 256, 20, 40), (46, 17, 30, 252, 227, 20, 40), (47, 30, 28, 373, 329, 20, 40), (48, 15, 35, 387, 386, 20, 40), (49, 25, 28, 387, 367, 20, 40), (50, 25, 27, 162, 121, 20, 40), (51, 28, 29, 162, 121, 20, 40), (52, 39, 10, 9, 6, 20, 40), (53, 25, 18, 41, 42, 20, 40), (54, 17, 10, 288, 297, 20, 40), (55, 23, 22, 50, 51, 20, 40), (56, 14, 16, 25, 28, 20, 40), (57, 23, 31, 264, 258, 20, 40), (58, 26, 35, 387, 385, 20, 40), (59, 36, 32, 230, 183, 20, 40), (60, 40, 29, 237, 212, 20, 40), (61, 21, 39, 302, 319, 20, 40), (62, 14, 36, 400, 380, 20, 40), (63, 15, 10, 271, 289, 20, 40), (64, 12, 33, 375, 377, 20, 40), (65, 37, 16, 167, 124, 20, 40), (66, 32, 25, 374, 366, 20, 40), (67, 24, 21, 357, 348, 20, 40), (68, 24, 31, 460, 424, 20, 40), (69, 23, 12, 297, 307, 20, 40), (70, 31, 40, 302, 289, 20, 40), (71, 14, 40, 429, 412, 20, 40), (72, 32, 16, 271, 256, 20, 40), (73, 23, 14, 271, 286, 20, 40), (74, 32, 19, 297, 259, 20, 40), (75, 40, 32, 429, 407, 20, 40), (76, 26, 34, 47, 28, 20, 40), (77, 18, 35, 53, 62, 20, 40), (78, 29, 18, 374, 351, 20, 40), (79, 36, 22, 259, 207, 20, 40), (80, 37, 30, 81, 70, 20, 40), (81, 15, 30, 429, 406, 20, 40), (82, 24, 38, 205, 204, 20, 40), (83, 13, 37, 81, 74, 20, 40), (84, 23, 38, 41, 17, 20, 40), (85, 10, 35, 302, 275, 20, 40), (86, 19, 35, 237, 212, 20, 40), (87, 24, 35, 193, 198, 20, 40), (88, 29, 23, 259, 253, 20, 40), (89, 28, 35, 9, 15, 20, 40), (90, 37, 10, 237, 227, 20, 40), (91, 29, 31, 321, 305, 20, 40), (92, 32, 33, 230, 200, 20, 40), (93, 40, 32, 395, 355, 20, 40), (94, 12, 11, 283, 269, 20, 40), (95, 33, 13, 47, 41, 20, 40), (96, 26, 37, 374, 332, 20, 40), (97, 22, 19, 400, 380, 20, 40), (98, 13, 33, 448, 419, 20, 40), (99, 18, 38, 191, 204, 20, 40), (100, 25, 38, 102, 78, 20, 40), (101, 38, 39, 221, 188, 20, 40), (102, 33, 21, 230, 204, 20, 40), (103, 30, 31, 321, 279, 20, 40), (104, 29, 19, 81, 73, 20, 40), (105, 17, 29, 373, 363, 20, 40), (106, 23, 12, 259, 231, 20, 40), (107, 38, 17, 193, 145, 20, 40), (108, 15, 40, 375, 375, 20, 40), (109, 28, 31, 81, 55, 20, 40), (110, 27, 38, 297, 299, 20, 40), (111, 18, 10, 321, 315, 20, 40), (112, 10, 26, 50, 29, 20, 40), (113, 11, 36, 167, 144, 20, 40), (114, 40, 20, 460, 411, 20, 40), (115, 12, 25, 47, 65, 20, 40), (116, 30, 11, 205, 196, 20, 40), (117, 27, 22, 191, 192, 20, 40), (118, 35, 27, 357, 333, 20, 40), (119, 14, 23, 288, 314, 20, 40), (120, 35, 37, 375, 365, 20, 40), (121, 36, 31, 395, 343, 20, 40), (122, 40, 33, 221, 221, 20, 40), (123, 37, 12, 50, 21, 20, 40), (124, 29, 26, 271, 232, 20, 40), (125, 35, 24, 375, 331, 20, 40), (126, 18, 36, 167, 153, 20, 40), (127, 16, 37, 191, 202, 20, 40), (128, 12, 19, 429, 414, 20, 40), (129, 35, 10, 283, 268, 20, 40), (130, 35, 12, 374, 370, 20, 40), (131, 27, 36, 440, 439, 20, 40), (132, 27, 18, 252, 210, 20, 40), (133, 25, 20, 269, 271, 20, 40), (134, 35, 12, 191, 184, 20, 40), (135, 31, 16, 387, 384, 20, 40), (136, 23, 34, 53, 26, 20, 40), (137, 14, 39, 373, 393, 20, 40), (138, 19, 18, 375, 355, 20, 40), (139, 16, 32, 302, 313, 20, 40), (140, 13, 22, 460, 448, 20, 40), (141, 39, 25, 191, 136, 20, 40), (142, 24, 21, 377, 382, 20, 40), (143, 20, 27, 283, 269, 20, 40), (144, 40, 20, 375, 347, 20, 40), (145, 14, 21, 25, 15, 20, 40), (146, 38, 24, 269, 239, 20, 40), (147, 17, 26, 69, 66, 20, 40), (148, 32, 30, 283, 269, 20, 40), (149, 35, 35, 41, 32, 20, 40), (150, 37, 31, 400, 392, 20, 40), (151, 29, 36, 402, 353, 20, 40), (152, 33, 12, 162, 118, 20, 40), (153, 39, 23, 221, 170, 20, 40), (154, 34, 20, 230, 208, 20, 40), (155, 11, 31, 374, 350, 20, 40), (156, 31, 39, 283, 292, 20, 40), (157, 18, 18, 373, 363, 20, 40), (158, 14, 15, 237, 256, 20, 40), (159, 17, 21, 297, 292, 20, 40), (160, 27, 24, 117, 110, 20, 40), (161, 24, 24, 448, 447, 20, 40), (162, 27, 40, 295, 279, 20, 40), (163, 16, 29, 193, 158, 20, 40), (164, 23, 13, 47, 26, 20, 40), (165, 29, 21, 429, 429, 20, 40), (166, 31, 30, 191, 143, 20, 40), (167, 29, 37, 264, 266, 20, 40), (168, 37, 22, 440, 402, 20, 40), (169, 36, 28, 41, 22, 20, 40), (170, 39, 23, 321, 282, 20, 40), (171, 32, 22, 400, 397, 20, 40), (172, 32, 24, 237, 230, 20, 40), (173, 36, 34, 374, 319, 20, 40), (174, 33, 39, 193, 142, 20, 40), (175, 33, 12, 387, 367, 20, 40), (176, 11, 26, 440, 468, 20, 40), (177, 40, 28, 357, 321, 20, 40), (178, 21, 37, 117, 133, 20, 40), (179, 15, 22, 374, 379, 20, 40), (180, 33, 16, 81, 71, 20, 40), (181, 17, 29, 340, 333, 20, 40), (182, 13, 22, 81, 82, 20, 40), (183, 30, 23, 357, 338, 20, 40), (184, 15, 28, 283, 273, 20, 40), (185, 37, 34, 302, 303, 20, 40), (186, 12, 11, 259, 257, 20, 40), (187, 27, 26, 221, 211, 20, 40), (188, 40, 17, 402, 373, 20, 40), (189, 16, 29, 374, 384, 20, 40), (190, 26, 15, 47, 40, 20, 40), (191, 32, 18, 252, 243, 20, 40), (192, 18, 34, 448, 410, 20, 40), (193, 24, 40, 377, 340, 20, 40), (194, 21, 30, 259, 265, 20, 40), (195, 28, 29, 162, 121, 20, 40), (196, 10, 40, 269, 243, 20, 40), (197, 10, 12, 377, 385, 20, 40), (198, 26, 26, 193, 150, 20, 40), (199, 22, 31, 47, 35, 20, 40), (200, 15, 15, 288, 312, 20, 40)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]

    # Set the constants
    c_t = 390
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_iterations = 740000  # Maximum number of iterations

    # Define grid search parameter values
    diversification_moves_list = [int(0.375 * len(jobs))]
    tenure_list = [int(0.25 * len(jobs))]

    # Running the grid search
    for diversification_moves in diversification_moves_list:
        for tenure in tenure_list:
            # Store results for the current configuration
            objective_values = []
            durations_list = []

            # Run the search 32 times, each with a different seed
            for i in range(32):
                start_time = time.time()

                # Run the local search for the given seed, diversification_moves, tenure, and max_iterations
                best_objective = simplified_random_local_search(jobs, c_t, max_time, seed_value=seeds[i], 
                                                                diversification_moves=diversification_moves, 
                                                                tenure=tenure, 
                                                                max_iterations=max_iterations)

                duration = time.time() - start_time

                # Skip the first run, which is used for JIT compilation
                if i > 1:
                    objective_values.append(best_objective)  # Capture best overall objective value
                    durations_list.append(duration)

            # Calculate summary statistics for the current configuration
            mean_objective = np.mean(objective_values)
            std_dev_objective = np.std(objective_values)
            min_objective = np.min(objective_values)
            max_objective = np.max(objective_values)
            q1_objective = np.percentile(objective_values, 25)
            q3_objective = np.percentile(objective_values, 75)
            average_duration = np.mean(durations_list)

            # Print the summary for the current configuration
            print(f"\nSummary for diversification_moves = {diversification_moves}, tenure = {tenure}:")
            print(f"Mean Objective Value: {mean_objective:.2f}")
            print(f"Standard Deviation: {std_dev_objective:.2f}")
            print(f"Minimum Objective Value: {min_objective:.2f}")
            print(f"Maximum Objective Value: {max_objective:.2f}")
            print(f"Q1 (25th percentile): {q1_objective:.2f}")
            print(f"Q3 (75th percentile): {q3_objective:.2f}")
            print(f"Average Duration per Run: {average_duration:.3f} seconds")


if __name__ == "__main__":
    main()