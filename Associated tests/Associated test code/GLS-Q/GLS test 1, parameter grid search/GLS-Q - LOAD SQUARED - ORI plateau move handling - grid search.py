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
    # Manually compute each quarter utility
    utilities_0 = load_at_quarter[0] / (1.0 + penalties[0])
    utilities_1 = load_at_quarter[1] / (1.0 + penalties[1])
    utilities_2 = load_at_quarter[2] / (1.0 + penalties[2])
    utilities_3 = load_at_quarter[3] / (1.0 + penalties[3])
    
    # Return the result as an array
    return np.array([utilities_0, utilities_1, utilities_2, utilities_3], dtype=np.float64)

@njit(cache=True)
def calculate_objective_value(total_overload, total_tardiness):
    return 10 * total_overload + total_tardiness

@njit(cache=True)
def calculate_augmented_objective(original_objective, load_at_time, penalties, nu, max_time):
    augmented_objective = original_objective
    quarter_size = max_time // 4

    # Handle first quarter
    for t in range(0, quarter_size):
        augmented_objective += nu * penalties[0] * (load_at_time[t] ** 2)

    # Handle second quarter
    for t in range(quarter_size, 2 * quarter_size):
        augmented_objective += nu * penalties[1] * (load_at_time[t] ** 2)

    # Handle third quarter
    for t in range(2 * quarter_size, 3 * quarter_size):
        augmented_objective += nu * penalties[2] * (load_at_time[t] ** 2)

    # Handle fourth quarter (including residual time periods)
    for t in range(3 * quarter_size, max_time + 1):
        augmented_objective += nu * penalties[3] * (load_at_time[t] ** 2)

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
    all_shifts = np.arange(-40, 41)
    plateau_moves = 0  # Initialize counter for plateau moves
    penalties = np.zeros(4, dtype=np.int32)  # Initialize penalty array
    best_original_objective = current_objective  # Track the best original objective value

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

            # Accept the move if the augmented objective improves or stays the same
            if augmented_objective <= current_objective:
                # Check if the original objective improves
                if original_objective < best_original_objective:
                    best_original_objective = original_objective
                    plateau_moves = 0  # Reset plateau moves if original objective improves
                else:
                    plateau_moves += 1  # Increment plateau moves if original objective does not improve

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
    jobs = np.array([(1, 33, 68, 50, 61, 40, 80), (2, 21, 54, 617, 619, 40, 80), (3, 66, 63, 95, 56, 40, 80), (4, 37, 71, 50, 39, 40, 80), (5, 64, 24, 732, 689, 40, 80), (6, 51, 80, 102, 116, 40, 80), (7, 32, 31, 598, 603, 40, 80), (8, 61, 53, 11, 21, 40, 80), (9, 68, 25, 587, 509, 40, 80), (10, 27, 66, 21, 9, 40, 80), (11, 42, 69, 655, 577, 40, 80), (12, 35, 79, 361, 341, 40, 80), (13, 37, 70, 755, 781, 40, 80), (14, 67, 76, 875, 833, 40, 80), (15, 56, 47, 433, 375, 40, 80), (16, 63, 21, 280, 293, 40, 80), (17, 74, 48, 920, 879, 40, 80), (18, 43, 35, 750, 743, 40, 80), (19, 70, 41, 50, 5, 40, 80), (20, 37, 28, 299, 249, 40, 80), (21, 46, 37, 910, 921, 40, 80), (22, 71, 32, 11, 11, 40, 80), (23, 50, 31, 724, 738, 40, 80), (24, 58, 58, 884, 902, 40, 80), (25, 26, 58, 280, 257, 40, 80), (26, 62, 38, 167, 102, 40, 80), (27, 49, 51, 109, 88, 40, 80), (28, 26, 54, 386, 428, 40, 80), (29, 62, 74, 413, 408, 40, 80), (30, 73, 32, 343, 314, 40, 80), (31, 25, 35, 167, 124, 40, 80), (32, 80, 52, 129, 45, 40, 80), (33, 26, 22, 588, 549, 40, 80), (34, 41, 47, 111, 90, 40, 80), (35, 70, 74, 571, 574, 40, 80), (36, 22, 53, 15, 50, 40, 80), (37, 76, 40, 884, 878, 40, 80), (38, 50, 24, 129, 132, 40, 80), (39, 66, 45, 682, 683, 40, 80), (40, 27, 75, 513, 556, 40, 80), (41, 24, 33, 655, 611, 40, 80), (42, 72, 70, 320, 289, 40, 80), (43, 58, 30, 95, 58, 40, 80), (44, 53, 51, 364, 353, 40, 80), (45, 40, 38, 221, 176, 40, 80), (46, 30, 50, 163, 131, 40, 80), (47, 54, 75, 102, 124, 40, 80), (48, 70, 74, 59, 32, 40, 80), (49, 44, 59, 907, 865, 40, 80), (50, 28, 50, 617, 572, 40, 80), (51, 31, 38, 682, 617, 40, 80), (52, 55, 54, 50, 22, 40, 80), (53, 39, 74, 786, 801, 40, 80), (54, 46, 55, 343, 351, 40, 80), (55, 70, 42, 273, 244, 40, 80), (56, 48, 72, 589, 557, 40, 80), (57, 60, 48, 128, 140, 40, 80), (58, 67, 65, 50, 11, 40, 80), (59, 70, 22, 96, 78, 40, 80), (60, 24, 65, 273, 233, 40, 80), (61, 21, 61, 364, 305, 40, 80), (62, 29, 72, 763, 751, 40, 80), (63, 77, 28, 131, 124, 40, 80), (64, 70, 55, 617, 536, 40, 80), (65, 59, 79, 212, 126, 40, 80), (66, 78, 50, 149, 86, 40, 80), (67, 47, 32, 587, 552, 40, 80), (68, 38, 73, 907, 860, 40, 80), (69, 79, 44, 21, 18, 40, 80), (70, 63, 42, 273, 249, 40, 80), (71, 79, 71, 511, 443, 40, 80), (72, 67, 29, 102, 66, 40, 80), (73, 45, 25, 167, 160, 40, 80), (74, 64, 38, 34, 29, 40, 80), (75, 56, 32, 647, 562, 40, 80), (76, 72, 72, 588, 551, 40, 80), (77, 31, 49, 11, 42, 40, 80), (78, 69, 37, 172, 87, 40, 80), (79, 44, 77, 299, 254, 40, 80), (80, 70, 66, 484, 388, 40, 80), (81, 27, 49, 587, 564, 40, 80), (82, 25, 45, 694, 645, 40, 80), (83, 26, 27, 637, 652, 40, 80), (84, 61, 32, 555, 541, 40, 80), (85, 26, 53, 50, 66, 40, 80), (86, 72, 28, 210, 162, 40, 80), (87, 24, 40, 212, 194, 40, 80), (88, 76, 78, 755, 682, 40, 80), (89, 78, 71, 198, 149, 40, 80), (90, 76, 41, 102, 69, 40, 80), (91, 66, 71, 98, 26, 40, 80), (92, 38, 69, 163, 176, 40, 80), (93, 31, 71, 755, 734, 40, 80), (94, 67, 39, 786, 797, 40, 80), (95, 62, 75, 804, 803, 40, 80), (96, 65, 41, 178, 188, 40, 80), (97, 50, 29, 21, 45, 40, 80), (98, 75, 40, 339, 335, 40, 80), (99, 28, 29, 124, 101, 40, 80), (100, 69, 47, 910, 856, 40, 80), (101, 48, 51, 574, 606, 40, 80), (102, 49, 55, 280, 277, 40, 80), (103, 76, 76, 320, 214, 40, 80), (104, 76, 21, 867, 834, 40, 80), (105, 54, 60, 320, 296, 40, 80), (106, 24, 46, 339, 364, 40, 80), (107, 32, 75, 373, 346, 40, 80), (108, 78, 36, 755, 723, 40, 80), (109, 76, 73, 786, 769, 40, 80), (110, 66, 33, 646, 609, 40, 80), (111, 53, 74, 128, 49, 40, 80), (112, 21, 73, 522, 580, 40, 80), (113, 25, 79, 197, 195, 40, 80), (114, 40, 52, 178, 206, 40, 80), (115, 80, 54, 198, 175, 40, 80), (116, 25, 72, 617, 588, 40, 80), (117, 48, 47, 320, 351, 40, 80), (118, 63, 64, 96, 15, 40, 80), (119, 25, 45, 647, 584, 40, 80), (120, 73, 73, 131, 72, 40, 80), (121, 39, 39, 98, 54, 40, 80), (122, 34, 59, 273, 227, 40, 80), (123, 67, 77, 193, 196, 40, 80), (124, 55, 46, 299, 224, 40, 80), (125, 72, 78, 11, 1, 40, 80), (126, 51, 45, 395, 367, 40, 80), (127, 57, 56, 149, 88, 40, 80), (128, 61, 29, 884, 816, 40, 80), (129, 68, 61, 178, 105, 40, 80), (130, 72, 25, 11, 18, 40, 80), (131, 69, 53, 655, 655, 40, 80), (132, 27, 64, 920, 964, 40, 80), (133, 39, 50, 163, 193, 40, 80), (134, 28, 58, 907, 933, 40, 80), (135, 44, 66, 637, 614, 40, 80), (136, 65, 50, 124, 92, 40, 80), (137, 38, 71, 844, 840, 40, 80), (138, 29, 77, 598, 536, 40, 80), (139, 35, 44, 907, 946, 40, 80), (140, 69, 74, 763, 710, 40, 80), (141, 78, 63, 867, 786, 40, 80), (142, 53, 20, 795, 722, 40, 80), (143, 30, 57, 713, 719, 40, 80), (144, 39, 44, 511, 479, 40, 80), (145, 53, 35, 779, 715, 40, 80), (146, 54, 25, 343, 283, 40, 80), (147, 48, 75, 646, 654, 40, 80), (148, 41, 28, 511, 538, 40, 80), (149, 58, 80, 907, 890, 40, 80), (150, 64, 41, 21, 7, 40, 80), (151, 46, 32, 910, 851, 40, 80), (152, 38, 38, 98, 98, 40, 80), (153, 38, 58, 34, 47, 40, 80), (154, 72, 24, 646, 596, 40, 80), (155, 42, 41, 779, 755, 40, 80), (156, 39, 32, 2, 36, 40, 80), (157, 20, 71, 364, 377, 40, 80), (158, 75, 22, 637, 531, 40, 80), (159, 69, 20, 795, 715, 40, 80), (160, 57, 39, 361, 356, 40, 80), (161, 46, 30, 779, 707, 40, 80), (162, 54, 73, 920, 914, 40, 80), (163, 32, 41, 451, 400, 40, 80), (164, 72, 37, 910, 898, 40, 80), (165, 37, 71, 598, 547, 40, 80), (166, 40, 72, 280, 288, 40, 80), (167, 60, 65, 721, 739, 40, 80), (168, 46, 49, 694, 676, 40, 80), (169, 35, 29, 750, 742, 40, 80), (170, 28, 64, 588, 591, 40, 80), (171, 33, 36, 124, 152, 40, 80), (172, 58, 53, 129, 86, 40, 80), (173, 24, 71, 395, 408, 40, 80), (174, 24, 66, 657, 701, 40, 80), (175, 78, 40, 910, 844, 40, 80), (176, 51, 76, 320, 244, 40, 80), (177, 20, 56, 273, 271, 40, 80), (178, 33, 73, 131, 98, 40, 80), (179, 46, 49, 750, 774, 40, 80), (180, 68, 20, 732, 707, 40, 80), (181, 53, 69, 555, 582, 40, 80), (182, 42, 46, 451, 394, 40, 80), (183, 20, 54, 724, 683, 40, 80), (184, 44, 56, 109, 122, 40, 80), (185, 47, 52, 617, 601, 40, 80), (186, 49, 42, 522, 440, 40, 80), (187, 72, 52, 173, 150, 40, 80), (188, 41, 75, 361, 386, 40, 80), (189, 49, 67, 364, 348, 40, 80), (190, 31, 20, 343, 349, 40, 80), (191, 43, 25, 95, 103, 40, 80), (192, 65, 61, 920, 925, 40, 80), (193, 44, 66, 386, 326, 40, 80), (194, 23, 65, 721, 775, 40, 80), (195, 75, 23, 285, 184, 40, 80), (196, 25, 21, 131, 166, 40, 80), (197, 70, 28, 124, 74, 40, 80), (198, 69, 79, 750, 698, 40, 80), (199, 69, 56, 682, 620, 40, 80), (200, 21, 62, 289, 249, 40, 80), (201, 54, 37, 285, 230, 40, 80), (202, 54, 35, 555, 553, 40, 80), (203, 74, 70, 555, 485, 40, 80), (204, 41, 56, 212, 142, 40, 80), (205, 31, 44, 364, 299, 40, 80), (206, 39, 39, 59, 15, 40, 80), (207, 68, 67, 2, 4, 40, 80), (208, 55, 39, 2, 14, 40, 80), (209, 73, 24, 339, 343, 40, 80), (210, 26, 40, 131, 96, 40, 80), (211, 67, 53, 779, 757, 40, 80), (212, 37, 23, 655, 637, 40, 80), (213, 39, 58, 149, 99, 40, 80), (214, 48, 65, 910, 851, 40, 80), (215, 31, 49, 484, 428, 40, 80), (216, 64, 31, 694, 657, 40, 80), (217, 28, 39, 289, 285, 40, 80), (218, 80, 48, 326, 272, 40, 80), (219, 55, 29, 212, 140, 40, 80), (220, 32, 58, 867, 891, 40, 80), (221, 76, 57, 867, 829, 40, 80), (222, 52, 22, 21, 36, 40, 80), (223, 36, 78, 59, 39, 40, 80), (224, 33, 24, 920, 946, 40, 80), (225, 75, 39, 646, 648, 40, 80), (226, 47, 27, 289, 301, 40, 80), (227, 44, 72, 289, 205, 40, 80), (228, 33, 35, 173, 140, 40, 80), (229, 44, 28, 395, 386, 40, 80), (230, 37, 35, 646, 636, 40, 80), (231, 32, 54, 167, 140, 40, 80), (232, 33, 49, 844, 781, 40, 80), (233, 29, 56, 197, 173, 40, 80), (234, 49, 41, 197, 164, 40, 80), (235, 62, 36, 221, 200, 40, 80), (236, 39, 42, 724, 735, 40, 80), (237, 40, 73, 907, 922, 40, 80), (238, 24, 49, 15, 31, 40, 80), (239, 76, 43, 280, 213, 40, 80), (240, 31, 24, 869, 906, 40, 80), (241, 56, 31, 910, 930, 40, 80), (242, 25, 53, 320, 358, 40, 80), (243, 38, 71, 750, 787, 40, 80), (244, 29, 42, 884, 886, 40, 80), (245, 75, 35, 920, 848, 40, 80), (246, 76, 44, 795, 711, 40, 80), (247, 56, 47, 167, 118, 40, 80), (248, 70, 53, 484, 468, 40, 80), (249, 24, 51, 869, 914, 40, 80), (250, 36, 54, 786, 735, 40, 80), (251, 43, 80, 364, 383, 40, 80), (252, 35, 49, 2, 2, 40, 80), (253, 66, 45, 875, 805, 40, 80), (254, 36, 59, 289, 280, 40, 80), (255, 60, 79, 59, 40, 40, 80), (256, 53, 52, 884, 883, 40, 80), (257, 44, 67, 96, 97, 40, 80), (258, 39, 44, 522, 451, 40, 80), (259, 54, 24, 721, 641, 40, 80), (260, 73, 22, 163, 163, 40, 80), (261, 41, 58, 844, 781, 40, 80), (262, 39, 76, 588, 573, 40, 80), (263, 64, 61, 750, 735, 40, 80), (264, 50, 56, 95, 28, 40, 80), (265, 73, 42, 694, 603, 40, 80), (266, 49, 38, 657, 685, 40, 80), (267, 51, 28, 910, 912, 40, 80), (268, 79, 39, 598, 526, 40, 80), (269, 48, 44, 763, 743, 40, 80), (270, 80, 21, 804, 701, 40, 80), (271, 63, 65, 732, 712, 40, 80), (272, 58, 47, 779, 694, 40, 80), (273, 60, 77, 875, 800, 40, 80), (274, 67, 49, 214, 222, 40, 80), (275, 50, 65, 682, 685, 40, 80), (276, 49, 25, 273, 210, 40, 80), (277, 76, 50, 867, 807, 40, 80), (278, 71, 73, 750, 724, 40, 80), (279, 27, 42, 574, 516, 40, 80), (280, 60, 35, 285, 239, 40, 80), (281, 58, 26, 157, 155, 40, 80), (282, 42, 57, 124, 145, 40, 80), (283, 20, 69, 163, 188, 40, 80), (284, 23, 40, 418, 407, 40, 80), (285, 63, 21, 124, 103, 40, 80), (286, 46, 37, 910, 888, 40, 80), (287, 20, 79, 299, 341, 40, 80), (288, 31, 57, 343, 323, 40, 80), (289, 47, 20, 214, 167, 40, 80), (290, 43, 38, 750, 670, 40, 80), (291, 56, 63, 361, 292, 40, 80), (292, 75, 65, 907, 907, 40, 80), (293, 37, 34, 320, 266, 40, 80), (294, 27, 64, 198, 232, 40, 80), (295, 72, 36, 193, 138, 40, 80), (296, 60, 24, 98, 75, 40, 80), (297, 27, 64, 339, 285, 40, 80), (298, 23, 71, 513, 539, 40, 80), (299, 71, 57, 102, 17, 40, 80), (300, 73, 20, 193, 134, 40, 80), (301, 36, 42, 78, 10, 40, 80), (302, 54, 45, 34, 52, 40, 80), (303, 71, 40, 589, 568, 40, 80), (304, 74, 77, 157, 141, 40, 80), (305, 62, 67, 361, 333, 40, 80), (306, 80, 78, 285, 167, 40, 80), (307, 51, 53, 78, 69, 40, 80), (308, 27, 38, 172, 223, 40, 80), (309, 38, 43, 221, 207, 40, 80), (310, 56, 30, 511, 457, 40, 80), (311, 43, 64, 173, 148, 40, 80), (312, 49, 29, 167, 193, 40, 80), (313, 45, 42, 844, 818, 40, 80), (314, 63, 76, 875, 774, 40, 80), (315, 27, 26, 682, 719, 40, 80), (316, 42, 48, 637, 654, 40, 80), (317, 62, 80, 713, 684, 40, 80), (318, 64, 28, 339, 278, 40, 80), (319, 26, 42, 157, 98, 40, 80), (320, 20, 28, 451, 504, 40, 80), (321, 34, 24, 157, 89, 40, 80), (322, 75, 54, 15, 17, 40, 80), (323, 71, 21, 11, 12, 40, 80), (324, 40, 27, 875, 883, 40, 80), (325, 76, 41, 869, 873, 40, 80), (326, 51, 79, 95, 90, 40, 80), (327, 63, 20, 109, 84, 40, 80), (328, 35, 20, 511, 487, 40, 80), (329, 76, 23, 589, 497, 40, 80), (330, 54, 80, 786, 732, 40, 80), (331, 25, 47, 214, 200, 40, 80), (332, 23, 59, 212, 213, 40, 80), (333, 44, 34, 637, 673, 40, 80), (334, 27, 25, 102, 141, 40, 80), (335, 78, 49, 193, 151, 40, 80), (336, 40, 24, 15, 30, 40, 80), (337, 61, 57, 59, 13, 40, 80), (338, 29, 31, 682, 623, 40, 80), (339, 48, 62, 513, 532, 40, 80), (340, 53, 50, 907, 840, 40, 80), (341, 22, 75, 541, 532, 40, 80), (342, 28, 54, 732, 773, 40, 80), (343, 43, 67, 59, 42, 40, 80), (344, 24, 79, 701, 652, 40, 80), (345, 41, 37, 484, 443, 40, 80), (346, 45, 27, 34, 29, 40, 80), (347, 32, 26, 433, 475, 40, 80), (348, 61, 63, 451, 401, 40, 80), (349, 20, 42, 221, 194, 40, 80), (350, 66, 56, 198, 124, 40, 80), (351, 72, 79, 721, 712, 40, 80), (352, 46, 40, 701, 645, 40, 80), (353, 46, 32, 128, 60, 40, 80), (354, 74, 53, 413, 419, 40, 80), (355, 49, 79, 128, 43, 40, 80), (356, 54, 23, 131, 102, 40, 80), (357, 79, 50, 285, 175, 40, 80), (358, 35, 21, 198, 217, 40, 80), (359, 39, 80, 11, 35, 40, 80), (360, 45, 30, 541, 520, 40, 80), (361, 44, 34, 617, 640, 40, 80), (362, 58, 26, 869, 802, 40, 80), (363, 35, 50, 721, 652, 40, 80), (364, 76, 47, 59, 49, 40, 80), (365, 40, 47, 713, 723, 40, 80), (366, 46, 20, 178, 196, 40, 80), (367, 67, 65, 221, 197, 40, 80), (368, 31, 37, 724, 707, 40, 80), (369, 68, 73, 149, 147, 40, 80), (370, 80, 42, 795, 737, 40, 80), (371, 30, 27, 285, 271, 40, 80), (372, 45, 60, 395, 368, 40, 80), (373, 38, 48, 221, 249, 40, 80), (374, 75, 57, 50, 41, 40, 80), (375, 25, 79, 867, 817, 40, 80), (376, 55, 58, 844, 844, 40, 80), (377, 45, 28, 173, 150, 40, 80), (378, 34, 76, 910, 873, 40, 80), (379, 21, 77, 418, 389, 40, 80), (380, 78, 79, 21, 12, 40, 80), (381, 29, 63, 326, 327, 40, 80), (382, 46, 79, 285, 234, 40, 80), (383, 20, 38, 221, 169, 40, 80), (384, 74, 79, 78, 72, 40, 80), (385, 58, 67, 59, 27, 40, 80), (386, 36, 58, 555, 489, 40, 80), (387, 79, 67, 779, 731, 40, 80), (388, 37, 74, 289, 313, 40, 80), (389, 79, 37, 157, 157, 40, 80), (390, 64, 21, 149, 58, 40, 80), (391, 24, 61, 598, 644, 40, 80), (392, 51, 32, 214, 132, 40, 80), (393, 32, 31, 289, 254, 40, 80), (394, 80, 68, 732, 658, 40, 80), (395, 63, 73, 724, 731, 40, 80), (396, 58, 43, 755, 723, 40, 80), (397, 59, 53, 646, 605, 40, 80), (398, 34, 44, 920, 853, 40, 80), (399, 60, 68, 102, 27, 40, 80), (400, 49, 69, 755, 781, 40, 80)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    c_t = 1800
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_iterations = 2200000
    max_stagnant_moves = 128000
    nu = 1  # Define the penalty multiplier nu
    
    results = []
    
    # Iterate over penalty reset thresholds from 2 to 20
    for penalty_reset_threshold in range(2, 20):
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