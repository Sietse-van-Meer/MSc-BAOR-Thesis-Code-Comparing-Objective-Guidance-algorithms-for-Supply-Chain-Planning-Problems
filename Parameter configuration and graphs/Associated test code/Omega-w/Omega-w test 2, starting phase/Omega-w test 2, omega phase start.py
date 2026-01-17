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
def calculate_augmented_objective_value(load_at_time, total_tardiness, c_t, nu):
    overload = np.maximum(0, load_at_time - c_t)
    total_overload = np.sum(overload)
    original_objective_value = calculate_objective_value(total_overload, total_tardiness)
    load_penalty = nu * np.sum(load_at_time ** 2)
    return original_objective_value +  load_penalty

@njit(cache=True)
def evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness, nu, phase):
    duration = job['processing_time']
    weight = job['weight']
    max_time = len(load_at_time) - 1 

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

    # If in augmented phase, calculate the augmented objective by adding the overload penalty
    if phase == 'augmented':
        augmented_objective = calculate_augmented_objective_value(load_at_time, total_tardiness + delta_tardiness, c_t, nu)
    else:
        augmented_objective = original_objective

    # Revert the temporary load adjustment (undo the changes)
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] -= weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] += weight

    return original_objective, augmented_objective, new_overload, delta_tardiness

@njit(cache=True)
def simplified_random_local_search(jobs, c_t, max_time, seed_value, max_plateau_moves, phase, start_iteration, nu, best_original_objective_value, max_iterations):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    load_at_time = calculate_load_at_time(current_solution, max_time)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    total_tardiness = np.sum(tardiness_values)
    total_overload = calculate_total_overload(load_at_time, c_t)

    # Track original and augmented objective values
    current_original_objective = calculate_objective_value(total_overload, total_tardiness)
    best_original_objective_value = current_original_objective  # Track best found original objective
    best_augmented_objective_value = calculate_augmented_objective_value(load_at_time, total_tardiness, c_t, nu) if phase == 'augmented' else np.nan

    job_indices = np.arange(len(jobs))
    all_shifts = np.arange(-20, 21)
    plateau_moves = 0  # Initialize counter for plateau moves
    iteration = start_iteration  # Initialize iteration counter

    while iteration < start_iteration + max_iterations and plateau_moves < max_plateau_moves:

        # Pick a random job and a random shift
        job_index = np.random.choice(job_indices)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(all_shifts)

        if shift == 0:
            continue

        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):  # Skip infeasible moves
            original_objective, augmented_objective, new_overload, delta_tardiness = evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness, nu, phase)

            # Update the current original objective after the move
            current_original_objective = calculate_objective_value(new_overload, total_tardiness + delta_tardiness)

            if phase == 'augmented':
                # For augmented phase: evaluate using augmented objective, but track plateau based on original objective
                if augmented_objective <= best_augmented_objective_value:
                    # Accept move if augmented objective improves or stays equal
                    best_augmented_objective_value = augmented_objective
                    
                    # Plateau moves are tracked based on original objective
                    if current_original_objective < best_original_objective_value:
                        plateau_moves = 0  # Improvement in original objective, reset plateau moves
                        best_original_objective_value = current_original_objective  # Update best original objective
                    else:
                        plateau_moves += 1  # No improvement or worsening in original objective
                    
                    # Make the move
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    total_tardiness += delta_tardiness
                    total_overload = new_overload
                    iteration += 1
                else:
                    # Reject the move if augmented objective worsens
                    plateau_moves += 1
                    iteration += 1
            else:
                # Normal phase: consider only the original objective for move acceptance and plateau tracking
                if original_objective < best_original_objective_value:
                    # Accept the move if original objective improves (goes down)
                    best_original_objective_value = original_objective
                    plateau_moves = 0  # Reset plateau moves on improvement
                    
                    # Make the move
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    total_tardiness += delta_tardiness
                    total_overload = new_overload
                    iteration += 1
                elif original_objective == best_original_objective_value:
                    # If original objective stays the same, increment plateau moves
                    plateau_moves += 1

                    # Make the move (since it's an equal objective value move)
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    total_tardiness += delta_tardiness
                    total_overload = new_overload
                    iteration += 1
                else:
                    # If original objective worsens, increment plateau moves and reject move
                    plateau_moves += 1
                    iteration += 1

            # Store iteration details including evaluated_value, best_original_objective_value, and min_objective_value


    # Instead of relying on the last iteration, return best_original_objective_value directly.
    return current_solution, best_original_objective_value

@njit(cache=True)
def local_search_phase_sequence(jobs, c_t, max_time, sequence, max_plateau_moves, nu, seed_value):
    current_solution = jobs.copy()
    cumulative_iteration = 0  # Track cumulative iterations
    best_objective_overall = np.inf  # Initialize to a large number to track the best

    for phase in sequence:
        # Modify to capture the best original objective value
        current_solution, best_original_objective_value = simplified_random_local_search(
            jobs=current_solution,
            c_t=c_t,
            max_time=max_time,
            seed_value=seed_value,
            max_plateau_moves=max_plateau_moves,
            phase=phase,  
            start_iteration=cumulative_iteration,
            nu=nu,
            best_original_objective_value = best_objective_overall,
            max_iterations=10000000000000  # Adjust as necessary
        )

        # Track the best objective value across all phases
        best_objective_overall = min(best_objective_overall, best_original_objective_value)

        if cumulative_iteration >= 10000000000000:
            break  # Terminate if max iterations reached or constraints violated

    # Return best overall objective and iteration details
    return current_solution, best_objective_overall


def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 22, 37, 221, 236, 20, 40), (2, 27, 40, 271, 247, 20, 40), (3, 28, 37, 253, 209, 20, 40), (4, 37, 18, 412, 360, 20, 40), (5, 15, 22, 42, 44, 20, 40), (6, 27, 11, 42, 37, 20, 40), (7, 22, 35, 460, 435, 20, 40), (8, 13, 30, 9, 30, 20, 40), (9, 16, 17, 141, 107, 20, 40), (10, 24, 38, 224, 233, 20, 40), (11, 12, 29, 148, 135, 20, 40), (12, 29, 27, 348, 330, 20, 40), (13, 25, 40, 412, 407, 20, 40), (14, 15, 33, 453, 447, 20, 40), (15, 39, 35, 271, 214, 20, 40), (16, 24, 35, 60, 58, 20, 40), (17, 10, 19, 26, 25, 20, 40), (18, 24, 16, 402, 368, 20, 40), (19, 11, 20, 220, 193, 20, 40), (20, 21, 36, 266, 259, 20, 40), (21, 22, 20, 110, 75, 20, 40), (22, 28, 17, 60, 16, 20, 40), (23, 28, 35, 378, 379, 20, 40), (24, 20, 22, 432, 410, 20, 40), (25, 35, 24, 304, 278, 20, 40), (26, 26, 29, 28, 42, 20, 40), (27, 24, 18, 453, 448, 20, 40), (28, 33, 22, 69, 32, 20, 40), (29, 11, 35, 214, 230, 20, 40), (30, 30, 29, 346, 306, 20, 40), (31, 27, 31, 448, 460, 20, 40), (32, 29, 24, 69, 41, 20, 40), (33, 33, 22, 82, 50, 20, 40), (34, 30, 34, 178, 188, 20, 40), (35, 28, 35, 296, 299, 20, 40), (36, 11, 29, 320, 299, 20, 40), (37, 35, 19, 449, 448, 20, 40), (38, 39, 18, 245, 223, 20, 40), (39, 31, 12, 415, 421, 20, 40), (40, 19, 38, 448, 451, 20, 40), (41, 30, 39, 53, 58, 20, 40), (42, 38, 17, 110, 78, 20, 40), (43, 19, 17, 195, 172, 20, 40), (44, 13, 24, 336, 356, 20, 40), (45, 18, 14, 141, 144, 20, 40), (46, 18, 11, 195, 160, 20, 40), (47, 29, 22, 361, 345, 20, 40), (48, 35, 22, 361, 327, 20, 40), (49, 28, 28, 57, 28, 20, 40), (50, 10, 14, 310, 287, 20, 40), (51, 38, 18, 408, 378, 20, 40), (52, 10, 22, 5, 14, 20, 40), (53, 17, 17, 336, 305, 20, 40), (54, 12, 39, 454, 446, 20, 40), (55, 34, 26, 385, 370, 20, 40), (56, 38, 36, 30, 19, 20, 40), (57, 12, 15, 346, 318, 20, 40), (58, 36, 37, 304, 259, 20, 40), (59, 28, 26, 57, 17, 20, 40), (60, 12, 10, 336, 310, 20, 40), (61, 33, 22, 49, 0, 20, 40), (62, 38, 30, 362, 346, 20, 40), (63, 30, 29, 362, 363, 20, 40), (64, 32, 13, 245, 241, 20, 40), (65, 19, 31, 148, 140, 20, 40), (66, 25, 35, 385, 370, 20, 40), (67, 36, 26, 348, 332, 20, 40), (68, 15, 33, 53, 63, 20, 40), (69, 22, 15, 454, 459, 20, 40), (70, 23, 23, 460, 424, 20, 40), (71, 33, 16, 432, 383, 20, 40), (72, 21, 27, 30, 16, 20, 40), (73, 31, 33, 354, 306, 20, 40), (74, 29, 18, 310, 304, 20, 40), (75, 30, 34, 411, 389, 20, 40), (76, 40, 14, 107, 85, 20, 40), (77, 17, 14, 69, 59, 20, 40), (78, 13, 40, 22, 25, 20, 40), (79, 37, 32, 110, 87, 20, 40), (80, 32, 36, 412, 364, 20, 40), (81, 15, 35, 245, 217, 20, 40), (82, 26, 21, 53, 48, 20, 40), (83, 28, 36, 266, 220, 20, 40), (84, 14, 39, 127, 98, 20, 40), (85, 28, 27, 338, 313, 20, 40), (86, 11, 17, 310, 321, 20, 40), (87, 11, 26, 53, 48, 20, 40), (88, 40, 21, 9, 5, 20, 40), (89, 34, 35, 310, 278, 20, 40), (90, 25, 10, 432, 417, 20, 40), (91, 22, 22, 271, 285, 20, 40), (92, 33, 15, 254, 210, 20, 40), (93, 39, 40, 348, 307, 20, 40), (94, 13, 39, 178, 171, 20, 40), (95, 32, 10, 170, 152, 20, 40), (96, 10, 22, 60, 60, 20, 40), (97, 38, 31, 5, 6, 20, 40), (98, 23, 35, 201, 186, 20, 40), (99, 25, 19, 221, 180, 20, 40), (100, 36, 30, 443, 411, 20, 40), (101, 28, 36, 253, 216, 20, 40), (102, 11, 19, 53, 82, 20, 40), (103, 39, 10, 76, 23, 20, 40), (104, 33, 24, 428, 430, 20, 40), (105, 39, 12, 373, 342, 20, 40), (106, 27, 15, 42, 35, 20, 40), (107, 36, 40, 110, 104, 20, 40), (108, 15, 20, 248, 257, 20, 40), (109, 38, 29, 28, 27, 20, 40), (110, 15, 29, 254, 261, 20, 40), (111, 18, 13, 201, 166, 20, 40), (112, 20, 23, 453, 463, 20, 40), (113, 11, 25, 195, 167, 20, 40), (114, 17, 20, 362, 359, 20, 40), (115, 14, 22, 127, 127, 20, 40), (116, 32, 12, 346, 334, 20, 40), (117, 32, 11, 53, 28, 20, 40), (118, 22, 17, 346, 321, 20, 40), (119, 13, 26, 141, 143, 20, 40), (120, 33, 40, 285, 240, 20, 40), (121, 10, 12, 68, 41, 20, 40), (122, 20, 28, 42, 30, 20, 40), (123, 22, 34, 248, 248, 20, 40), (124, 27, 27, 76, 70, 20, 40), (125, 33, 22, 362, 342, 20, 40), (126, 16, 33, 361, 327, 20, 40), (127, 19, 12, 107, 77, 20, 40), (128, 34, 22, 412, 404, 20, 40), (129, 37, 14, 408, 389, 20, 40), (130, 17, 32, 19, 24, 20, 40), (131, 37, 18, 443, 433, 20, 40), (132, 21, 40, 134, 133, 20, 40), (133, 25, 35, 361, 370, 20, 40), (134, 10, 31, 107, 104, 20, 40), (135, 38, 20, 443, 391, 20, 40), (136, 18, 16, 269, 263, 20, 40), (137, 15, 29, 373, 376, 20, 40), (138, 14, 19, 314, 325, 20, 40), (139, 32, 16, 69, 30, 20, 40), (140, 25, 16, 408, 403, 20, 40), (141, 10, 12, 221, 203, 20, 40), (142, 15, 12, 49, 45, 20, 40), (143, 11, 30, 248, 235, 20, 40), (144, 36, 40, 320, 298, 20, 40), (145, 25, 37, 443, 417, 20, 40), (146, 24, 27, 254, 238, 20, 40), (147, 14, 14, 220, 241, 20, 40), (148, 37, 23, 402, 400, 20, 40), (149, 21, 40, 412, 424, 20, 40), (150, 39, 16, 50, 47, 20, 40), (151, 17, 18, 68, 40, 20, 40), (152, 28, 27, 127, 109, 20, 40), (153, 19, 36, 214, 187, 20, 40), (154, 33, 20, 443, 437, 20, 40), (155, 29, 16, 49, 48, 20, 40), (156, 27, 18, 408, 399, 20, 40), (157, 38, 34, 453, 442, 20, 40), (158, 30, 21, 19, 13, 20, 40), (159, 11, 35, 201, 202, 20, 40), (160, 20, 35, 53, 17, 20, 40), (161, 22, 30, 402, 402, 20, 40), (162, 14, 19, 256, 226, 20, 40), (163, 11, 17, 328, 345, 20, 40), (164, 13, 13, 428, 414, 20, 40), (165, 17, 24, 53, 49, 20, 40), (166, 28, 31, 201, 163, 20, 40), (167, 15, 24, 385, 371, 20, 40), (168, 16, 26, 195, 178, 20, 40), (169, 30, 26, 26, 15, 20, 40), (170, 20, 37, 57, 42, 20, 40), (171, 16, 25, 224, 188, 20, 40), (172, 40, 11, 296, 256, 20, 40), (173, 32, 20, 296, 260, 20, 40), (174, 36, 16, 449, 426, 20, 40), (175, 11, 25, 82, 54, 20, 40), (176, 38, 29, 69, 35, 20, 40), (177, 25, 34, 224, 207, 20, 40), (178, 39, 31, 338, 294, 20, 40), (179, 29, 33, 57, 18, 20, 40), (180, 13, 22, 214, 211, 20, 40), (181, 39, 18, 460, 436, 20, 40), (182, 17, 39, 276, 282, 20, 40), (183, 24, 40, 5, 21, 20, 40), (184, 36, 40, 9, 6, 20, 40), (185, 17, 16, 49, 32, 20, 40), (186, 23, 40, 408, 402, 20, 40), (187, 23, 21, 134, 140, 20, 40), (188, 20, 21, 354, 356, 20, 40), (189, 14, 19, 361, 374, 20, 40), (190, 13, 14, 22, 42, 20, 40), (191, 19, 22, 428, 415, 20, 40), (192, 27, 36, 354, 314, 20, 40), (193, 21, 39, 285, 244, 20, 40), (194, 33, 13, 304, 255, 20, 40), (195, 33, 19, 420, 382, 20, 40), (196, 24, 30, 304, 311, 20, 40), (197, 10, 38, 60, 90, 20, 40), (198, 33, 32, 224, 180, 20, 40), (199, 34, 16, 320, 287, 20, 40), (200, 10, 24, 148, 158, 20, 40)

], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    sequences = [['normal'], ['normal', 'augmented'], ['normal', 'augmented', 'normal'], ['normal', 'augmented', 'normal', 'augmented'], ['normal', 'augmented', 'normal', 'augmented', 'normal']]
    nu = 1  # Overload penalty multiplier
    c_t = 140  # Capacity threshold
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_plateau_moves =  32000
    
    for sequence in sequences:

        # Store results for each run
        objective_values = []
        durations_list = []

        # Running the search 32 times, calculating summary after 32 runs
        for i in range(32):
            start_time = time.time()

            # Run the local search for the given sequence
            solution, best_objective_overall = local_search_phase_sequence(
                jobs, c_t, max_time, sequence, max_plateau_moves, nu, seed_value=seeds[i]
            )

            duration = time.time() - start_time

            # Skip the first run, which is used for JIT compilation
            if i > 1:
                objective_values.append(best_objective_overall)  # Capture best overall objective value
                durations_list.append(duration)

        # Calculate summary statistics
        mean_objective = np.mean(objective_values)
        std_dev_objective = np.std(objective_values)
        min_objective = np.min(objective_values)
        max_objective = np.max(objective_values)
        q1_objective = np.percentile(objective_values, 25)
        q3_objective = np.percentile(objective_values, 75)

        average_duration = np.mean(durations_list)

        # Print summary
        print(f"\nSummary of Objective Values over 30 runs for sequence {sequence}:")
        print(f"Mean Objective Value: {mean_objective:.2f}")
        print(f"Standard Deviation: {std_dev_objective:.2f}")
        print(f"Minimum Objective Value: {min_objective:.2f}")
        print(f"Maximum Objective Value: {max_objective:.2f}")
        print(f"Q1 (25th percentile): {q1_objective:.2f}")
        print(f"Q3 (75th percentile): {q3_objective:.2f}")
        print(f"Average Duration per Run: {average_duration:.3f} seconds")


if __name__ == "__main__":
    main()