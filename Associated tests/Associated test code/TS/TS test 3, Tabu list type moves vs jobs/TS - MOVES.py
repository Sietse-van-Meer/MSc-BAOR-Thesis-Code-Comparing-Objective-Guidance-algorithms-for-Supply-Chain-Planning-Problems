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
    jobs = np.array([(1, 76, 28, 15, 7, 40, 80), (2, 32, 29, 0, 17, 40, 80), (3, 59, 31, 30, 47, 40, 80), (4, 30, 61, 456, 417, 40, 80), (5, 70, 41, 417, 372, 40, 80), (6, 70, 36, 90, 68, 40, 80), (7, 78, 21, 256, 155, 40, 80), (8, 52, 76, 296, 247, 40, 80), (9, 65, 72, 274, 186, 40, 80), (10, 61, 79, 197, 138, 40, 80), (11, 48, 31, 167, 146, 40, 80), (12, 55, 34, 248, 170, 40, 80), (13, 39, 56, 274, 293, 40, 80), (14, 38, 72, 324, 259, 40, 80), (15, 26, 77, 62, 34, 40, 80), (16, 71, 51, 229, 210, 40, 80), (17, 63, 55, 366, 281, 40, 80), (18, 69, 67, 221, 152, 40, 80), (19, 46, 71, 128, 46, 40, 80), (20, 71, 28, 253, 245, 40, 80), (21, 70, 56, 446, 361, 40, 80), (22, 51, 49, 180, 128, 40, 80), (23, 27, 37, 462, 404, 40, 80), (24, 78, 55, 38, 39, 40, 80), (25, 28, 38, 96, 51, 40, 80), (26, 25, 45, 406, 391, 40, 80), (27, 32, 55, 21, 8, 40, 80), (28, 80, 26, 214, 205, 40, 80), (29, 52, 63, 184, 122, 40, 80), (30, 61, 35, 313, 240, 40, 80), (31, 32, 25, 314, 362, 40, 80), (32, 34, 77, 158, 170, 40, 80), (33, 57, 73, 520, 517, 40, 80), (34, 63, 41, 19, 8, 40, 80), (35, 69, 62, 107, 97, 40, 80), (36, 46, 80, 219, 181, 40, 80), (37, 26, 63, 77, 98, 40, 80), (38, 47, 55, 31, 7, 40, 80), (39, 79, 53, 262, 213, 40, 80), (40, 54, 75, 500, 445, 40, 80), (41, 27, 38, 43, 16, 40, 80), (42, 71, 47, 128, 114, 40, 80), (43, 31, 32, 149, 139, 40, 80), (44, 68, 28, 229, 237, 40, 80), (45, 71, 62, 406, 410, 40, 80), (46, 75, 68, 462, 374, 40, 80), (47, 49, 74, 344, 295, 40, 80), (48, 61, 38, 180, 87, 40, 80), (49, 31, 68, 314, 337, 40, 80), (50, 21, 64, 511, 511, 40, 80), (51, 47, 77, 316, 288, 40, 80), (52, 26, 32, 119, 65, 40, 80), (53, 72, 62, 194, 196, 40, 80), (54, 37, 35, 219, 241, 40, 80), (55, 51, 28, 493, 522, 40, 80), (56, 23, 65, 120, 151, 40, 80), (57, 69, 20, 253, 187, 40, 80), (58, 52, 54, 199, 185, 40, 80), (59, 57, 56, 306, 214, 40, 80), (60, 64, 51, 21, 11, 40, 80), (61, 79, 42, 197, 112, 40, 80), (62, 67, 35, 146, 92, 40, 80), (63, 29, 34, 229, 232, 40, 80), (64, 32, 71, 146, 150, 40, 80), (65, 67, 21, 21, 29, 40, 80), (66, 52, 68, 149, 132, 40, 80), (67, 68, 45, 291, 279, 40, 80), (68, 28, 22, 230, 250, 40, 80), (69, 56, 77, 15, 38, 40, 80), (70, 41, 69, 156, 140, 40, 80), (71, 66, 36, 128, 72, 40, 80), (72, 67, 45, 120, 83, 40, 80), (73, 71, 24, 156, 164, 40, 80), (74, 56, 77, 82, 105, 40, 80), (75, 60, 34, 100, 79, 40, 80), (76, 28, 64, 248, 237, 40, 80), (77, 67, 23, 199, 110, 40, 80), (78, 40, 37, 43, 45, 40, 80), (79, 49, 73, 194, 161, 40, 80), (80, 70, 24, 324, 245, 40, 80), (81, 21, 23, 462, 434, 40, 80), (82, 72, 24, 423, 330, 40, 80), (83, 73, 71, 6, 11, 40, 80), (84, 66, 69, 314, 304, 40, 80), (85, 42, 39, 144, 120, 40, 80), (86, 65, 33, 329, 272, 40, 80), (87, 55, 50, 19, 19, 40, 80), (88, 47, 79, 433, 418, 40, 80), (89, 66, 57, 471, 463, 40, 80), (90, 21, 26, 225, 189, 40, 80), (91, 52, 33, 359, 328, 40, 80), (92, 73, 32, 262, 248, 40, 80), (93, 34, 66, 174, 119, 40, 80), (94, 80, 34, 296, 228, 40, 80), (95, 61, 36, 439, 338, 40, 80), (96, 66, 74, 229, 231, 40, 80), (97, 38, 77, 189, 135, 40, 80), (98, 47, 45, 410, 360, 40, 80), (99, 52, 28, 314, 228, 40, 80), (100, 33, 41, 158, 192, 40, 80), (101, 64, 44, 65, 3, 40, 80), (102, 60, 49, 439, 382, 40, 80), (103, 35, 64, 471, 442, 40, 80), (104, 47, 67, 162, 102, 40, 80), (105, 35, 26, 316, 268, 40, 80), (106, 78, 26, 450, 405, 40, 80), (107, 29, 23, 291, 291, 40, 80), (108, 80, 32, 367, 256, 40, 80), (109, 36, 63, 144, 128, 40, 80), (110, 34, 35, 262, 306, 40, 80), (111, 67, 38, 292, 281, 40, 80), (112, 30, 20, 119, 76, 40, 80), (113, 46, 67, 120, 70, 40, 80), (114, 60, 37, 469, 379, 40, 80), (115, 71, 67, 31, 13, 40, 80), (116, 37, 26, 434, 451, 40, 80), (117, 49, 43, 107, 56, 40, 80), (118, 38, 68, 134, 138, 40, 80), (119, 74, 65, 30, 24, 40, 80), (120, 66, 40, 433, 430, 40, 80), (121, 49, 22, 487, 461, 40, 80), (122, 69, 28, 493, 443, 40, 80), (123, 59, 72, 134, 80, 40, 80), (124, 43, 36, 520, 465, 40, 80), (125, 59, 73, 230, 206, 40, 80), (126, 35, 24, 134, 109, 40, 80), (127, 59, 32, 253, 158, 40, 80), (128, 64, 51, 406, 356, 40, 80), (129, 27, 36, 77, 38, 40, 80), (130, 22, 55, 6, 54, 40, 80), (131, 48, 35, 434, 419, 40, 80), (132, 72, 31, 313, 235, 40, 80), (133, 41, 67, 324, 337, 40, 80), (134, 53, 24, 453, 450, 40, 80), (135, 80, 27, 359, 265, 40, 80), (136, 64, 38, 201, 185, 40, 80), (137, 26, 49, 123, 62, 40, 80), (138, 36, 71, 31, 36, 40, 80), (139, 65, 46, 144, 55, 40, 80), (140, 21, 38, 248, 219, 40, 80), (141, 28, 58, 488, 429, 40, 80), (142, 60, 65, 356, 302, 40, 80), (143, 60, 46, 356, 267, 40, 80), (144, 43, 45, 82, 51, 40, 80), (145, 28, 49, 296, 335, 40, 80), (146, 41, 64, 50, 48, 40, 80), (147, 49, 35, 308, 302, 40, 80), (148, 40, 30, 367, 358, 40, 80), (149, 67, 22, 240, 170, 40, 80), (150, 77, 63, 417, 351, 40, 80), (151, 32, 64, 128, 118, 40, 80), (152, 57, 72, 50, 40, 40, 80), (153, 58, 77, 211, 232, 40, 80), (154, 37, 68, 423, 440, 40, 80), (155, 79, 60, 367, 307, 40, 80), (156, 33, 48, 162, 92, 40, 80), (157, 28, 32, 465, 466, 40, 80), (158, 53, 79, 410, 328, 40, 80), (159, 58, 62, 180, 104, 40, 80), (160, 22, 21, 493, 484, 40, 80), (161, 59, 66, 15, 17, 40, 80), (162, 43, 67, 25, 31, 40, 80), (163, 70, 53, 291, 244, 40, 80), (164, 29, 61, 469, 425, 40, 80), (165, 49, 37, 316, 335, 40, 80), (166, 49, 29, 30, 41, 40, 80), (167, 20, 80, 367, 390, 40, 80), (168, 43, 77, 128, 132, 40, 80), (169, 80, 41, 198, 196, 40, 80), (170, 55, 39, 123, 99, 40, 80), (171, 45, 53, 491, 414, 40, 80), (172, 33, 43, 313, 286, 40, 80), (173, 23, 77, 123, 127, 40, 80), (174, 51, 38, 198, 135, 40, 80), (175, 69, 42, 387, 338, 40, 80), (176, 20, 68, 41, 28, 40, 80), (177, 41, 63, 194, 221, 40, 80), (178, 45, 24, 25, 33, 40, 80), (179, 66, 43, 263, 251, 40, 80), (180, 77, 77, 307, 261, 40, 80), (181, 46, 57, 6, 28, 40, 80), (182, 53, 44, 462, 397, 40, 80), (183, 35, 50, 214, 169, 40, 80), (184, 34, 40, 65, 20, 40, 80), (185, 45, 37, 456, 483, 40, 80), (186, 65, 44, 197, 184, 40, 80), (187, 33, 43, 291, 268, 40, 80), (188, 31, 51, 308, 332, 40, 80), (189, 33, 69, 487, 463, 40, 80), (190, 32, 56, 520, 542, 40, 80), (191, 33, 36, 77, 83, 40, 80), (192, 55, 42, 456, 479, 40, 80), (193, 34, 58, 201, 234, 40, 80), (194, 63, 74, 146, 43, 40, 80), (195, 73, 50, 149, 115, 40, 80), (196, 43, 41, 120, 141, 40, 80), (197, 75, 79, 184, 159, 40, 80), (198, 73, 72, 410, 377, 40, 80), (199, 60, 43, 198, 155, 40, 80), (200, 61, 42, 15, 7, 40, 80), (201, 45, 55, 181, 183, 40, 80), (202, 48, 60, 274, 190, 40, 80), (203, 66, 24, 119, 34, 40, 80), (204, 71, 24, 49, 28, 40, 80), (205, 77, 34, 123, 6, 40, 80), (206, 71, 56, 248, 158, 40, 80), (207, 48, 46, 446, 412, 40, 80), (208, 44, 62, 56, 4, 40, 80), (209, 63, 60, 181, 104, 40, 80), (210, 52, 33, 499, 496, 40, 80), (211, 51, 48, 80, 50, 40, 80), (212, 50, 38, 411, 397, 40, 80), (213, 56, 21, 415, 394, 40, 80), (214, 60, 28, 423, 333, 40, 80), (215, 35, 47, 199, 192, 40, 80), (216, 57, 68, 292, 252, 40, 80), (217, 68, 45, 359, 359, 40, 80), (218, 51, 40, 366, 383, 40, 80), (219, 24, 79, 156, 144, 40, 80), (220, 28, 47, 211, 234, 40, 80), (221, 49, 43, 15, 1, 40, 80), (222, 76, 21, 144, 132, 40, 80), (223, 29, 38, 314, 319, 40, 80), (224, 30, 35, 189, 200, 40, 80), (225, 54, 72, 194, 170, 40, 80), (226, 51, 47, 439, 451, 40, 80), (227, 61, 80, 324, 288, 40, 80), (228, 33, 48, 500, 454, 40, 80), (229, 53, 26, 248, 179, 40, 80), (230, 64, 32, 291, 188, 40, 80), (231, 69, 34, 291, 243, 40, 80), (232, 38, 42, 356, 286, 40, 80), (233, 31, 46, 123, 170, 40, 80), (234, 46, 35, 238, 153, 40, 80), (235, 67, 23, 161, 141, 40, 80), (236, 44, 77, 149, 97, 40, 80), (237, 65, 25, 181, 149, 40, 80), (238, 66, 68, 224, 224, 40, 80), (239, 72, 32, 128, 18, 40, 80), (240, 68, 44, 274, 173, 40, 80), (241, 66, 61, 248, 149, 40, 80), (242, 80, 71, 41, 30, 40, 80), (243, 73, 74, 31, 30, 40, 80), (244, 30, 50, 214, 247, 40, 80), (245, 64, 28, 415, 387, 40, 80), (246, 25, 75, 100, 53, 40, 80), (247, 64, 78, 262, 275, 40, 80), (248, 67, 35, 161, 109, 40, 80), (249, 21, 42, 411, 375, 40, 80), (250, 20, 25, 433, 460, 40, 80), (251, 53, 21, 201, 156, 40, 80), (252, 72, 52, 493, 424, 40, 80), (253, 62, 46, 423, 388, 40, 80), (254, 78, 36, 185, 164, 40, 80), (255, 38, 60, 323, 265, 40, 80), (256, 65, 57, 433, 380, 40, 80), (257, 53, 64, 278, 198, 40, 80), (258, 73, 22, 199, 141, 40, 80), (259, 39, 41, 520, 488, 40, 80), (260, 21, 36, 30, 38, 40, 80), (261, 53, 58, 77, 11, 40, 80), (262, 64, 62, 411, 403, 40, 80), (263, 70, 49, 469, 452, 40, 80), (264, 45, 77, 144, 132, 40, 80), (265, 66, 23, 278, 182, 40, 80), (266, 23, 45, 221, 220, 40, 80), (267, 58, 46, 344, 267, 40, 80), (268, 47, 35, 387, 377, 40, 80), (269, 59, 58, 225, 150, 40, 80), (270, 65, 78, 50, 55, 40, 80), (271, 47, 37, 228, 175, 40, 80), (272, 73, 52, 103, 33, 40, 80), (273, 70, 32, 296, 224, 40, 80), (274, 25, 38, 292, 271, 40, 80), (275, 39, 66, 445, 466, 40, 80), (276, 20, 36, 387, 333, 40, 80), (277, 24, 62, 198, 249, 40, 80), (278, 60, 31, 316, 281, 40, 80), (279, 48, 26, 423, 410, 40, 80), (280, 43, 80, 161, 182, 40, 80), (281, 72, 45, 161, 76, 40, 80), (282, 20, 59, 11, 18, 40, 80), (283, 61, 53, 31, 42, 40, 80), (284, 33, 65, 411, 437, 40, 80), (285, 37, 45, 49, 8, 40, 80), (286, 57, 28, 423, 352, 40, 80), (287, 45, 22, 189, 211, 40, 80), (288, 35, 53, 367, 296, 40, 80), (289, 70, 30, 499, 435, 40, 80), (290, 59, 71, 199, 199, 40, 80), (291, 52, 50, 316, 253, 40, 80), (292, 79, 75, 274, 188, 40, 80), (293, 50, 55, 189, 147, 40, 80), (294, 71, 79, 174, 169, 40, 80), (295, 38, 54, 292, 268, 40, 80), (296, 39, 26, 199, 169, 40, 80), (297, 44, 64, 199, 179, 40, 80), (298, 24, 53, 471, 451, 40, 80), (299, 71, 78, 456, 421, 40, 80), (300, 20, 66, 296, 259, 40, 80), (301, 41, 29, 324, 266, 40, 80), (302, 25, 48, 194, 189, 40, 80), (303, 20, 23, 103, 71, 40, 80), (304, 32, 45, 473, 469, 40, 80), (305, 61, 50, 456, 400, 40, 80), (306, 72, 79, 120, 40, 40, 80), (307, 71, 37, 90, 72, 40, 80), (308, 38, 60, 445, 466, 40, 80), (309, 24, 63, 158, 188, 40, 80), (310, 23, 80, 198, 255, 40, 80), (311, 73, 48, 211, 194, 40, 80), (312, 59, 43, 158, 160, 40, 80), (313, 55, 55, 278, 294, 40, 80), (314, 38, 48, 445, 418, 40, 80), (315, 76, 80, 221, 203, 40, 80), (316, 71, 22, 253, 220, 40, 80), (317, 66, 59, 6, 4, 40, 80), (318, 52, 67, 128, 55, 40, 80), (319, 66, 73, 445, 354, 40, 80), (320, 43, 55, 146, 79, 40, 80), (321, 22, 61, 308, 351, 40, 80), (322, 38, 65, 181, 111, 40, 80), (323, 69, 74, 0, 3, 40, 80), (324, 57, 34, 103, 56, 40, 80), (325, 74, 52, 149, 113, 40, 80), (326, 41, 71, 471, 402, 40, 80), (327, 26, 75, 520, 533, 40, 80), (328, 49, 64, 500, 489, 40, 80), (329, 58, 73, 43, 10, 40, 80), (330, 37, 28, 134, 175, 40, 80), (331, 66, 28, 224, 171, 40, 80), (332, 43, 66, 366, 369, 40, 80), (333, 49, 21, 256, 230, 40, 80), (334, 67, 37, 253, 230, 40, 80), (335, 35, 72, 453, 386, 40, 80), (336, 68, 64, 146, 152, 40, 80), (337, 45, 69, 229, 179, 40, 80), (338, 75, 77, 499, 481, 40, 80), (339, 26, 56, 469, 498, 40, 80), (340, 30, 36, 49, 53, 40, 80), (341, 31, 40, 229, 202, 40, 80), (342, 22, 67, 238, 219, 40, 80), (343, 41, 67, 61, 58, 40, 80), (344, 43, 28, 453, 377, 40, 80), (345, 69, 42, 61, 40, 40, 80), (346, 71, 51, 411, 374, 40, 80), (347, 26, 27, 62, 64, 40, 80), (348, 58, 26, 520, 501, 40, 80), (349, 54, 59, 185, 99, 40, 80), (350, 44, 68, 6, 0, 40, 80), (351, 50, 76, 433, 435, 40, 80), (352, 30, 73, 278, 303, 40, 80), (353, 24, 45, 198, 252, 40, 80), (354, 79, 41, 197, 139, 40, 80), (355, 26, 23, 194, 199, 40, 80), (356, 54, 57, 511, 535, 40, 80), (357, 49, 22, 253, 229, 40, 80), (358, 50, 80, 469, 483, 40, 80), (359, 36, 52, 306, 240, 40, 80), (360, 41, 27, 167, 124, 40, 80), (361, 62, 70, 515, 458, 40, 80), (362, 56, 24, 491, 405, 40, 80), (363, 55, 69, 31, 49, 40, 80), (364, 54, 41, 359, 306, 40, 80), (365, 78, 29, 144, 92, 40, 80), (366, 34, 61, 417, 437, 40, 80), (367, 35, 42, 29, 66, 40, 80), (368, 68, 67, 274, 180, 40, 80), (369, 41, 34, 339, 353, 40, 80), (370, 69, 41, 144, 101, 40, 80), (371, 60, 79, 43, 16, 40, 80), (372, 80, 55, 433, 421, 40, 80), (373, 53, 61, 406, 343, 40, 80), (374, 52, 24, 248, 204, 40, 80), (375, 76, 69, 197, 132, 40, 80), (376, 63, 29, 50, 18, 40, 80), (377, 30, 53, 151, 158, 40, 80), (378, 23, 30, 90, 118, 40, 80), (379, 72, 60, 465, 445, 40, 80), (380, 51, 80, 462, 410, 40, 80), (381, 21, 33, 493, 530, 40, 80), (382, 39, 59, 278, 219, 40, 80), (383, 70, 21, 21, 11, 40, 80), (384, 75, 45, 411, 358, 40, 80), (385, 68, 37, 189, 102, 40, 80), (386, 31, 35, 324, 343, 40, 80), (387, 46, 71, 197, 202, 40, 80), (388, 75, 44, 410, 387, 40, 80), (389, 37, 55, 224, 242, 40, 80), (390, 67, 24, 224, 194, 40, 80), (391, 43, 63, 417, 395, 40, 80), (392, 79, 56, 308, 218, 40, 80), (393, 25, 26, 19, 66, 40, 80), (394, 65, 28, 240, 254, 40, 80), (395, 26, 43, 65, 109, 40, 80), (396, 28, 42, 439, 490, 40, 80), (397, 67, 72, 366, 309, 40, 80), (398, 30, 45, 38, 64, 40, 80), (399, 37, 43, 359, 297, 40, 80), (400, 58, 33, 307, 316, 40, 80)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]

    c_t = 2400
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    diversification_moves = int(0.375 * len(jobs))
    tenure = int(0.15 * len(jobs))  
    max_iterations_list = [2000000, 2700000,3600000, 4400000,5500000]  # Maximum number of iterations

    # Running the search for each value of max_iterations
    for max_iterations in max_iterations_list:
        # Store results for the current max_iterations
        objective_values = []
        durations_list = []

        # Run the search 32 times, each with a different seed
        for i in range(32):
            start_time = time.time()

            # Run the local search for the given seed and max_iterations
            best_objective = simplified_random_local_search(jobs, c_t, max_time, seed_value=seeds[i], 
                                                            diversification_moves=diversification_moves, 
                                                            tenure=tenure, 
                                                            max_iterations=max_iterations)

            duration = time.time() - start_time

            # Skip the first run, which is used for JIT compilation
            if i > 1:
                objective_values.append(best_objective)  # Capture best overall objective value
                durations_list.append(duration)

        # Calculate summary statistics for the current max_iterations
        mean_objective = np.mean(objective_values)
        std_dev_objective = np.std(objective_values)
        min_objective = np.min(objective_values)
        max_objective = np.max(objective_values)
        q1_objective = np.percentile(objective_values, 25)
        q3_objective = np.percentile(objective_values, 75)
        average_duration = np.mean(durations_list)

        # Print the summary for this max_iterations value
        print(f"\nSummary of Objective Values for max_iterations = {max_iterations}:")
        print(f"Mean Objective Value: {mean_objective:.2f}")
        print(f"Standard Deviation: {std_dev_objective:.2f}")
        print(f"Minimum Objective Value: {min_objective:.2f}")
        print(f"Maximum Objective Value: {max_objective:.2f}")
        print(f"Q1 (25th percentile): {q1_objective:.2f}")
        print(f"Q3 (75th percentile): {q3_objective:.2f}")
        print(f"Average Duration per Run: {average_duration:.3f} seconds")

if __name__ == "__main__":
    main()
