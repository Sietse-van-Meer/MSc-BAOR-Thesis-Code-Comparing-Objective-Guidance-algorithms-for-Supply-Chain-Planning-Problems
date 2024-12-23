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
def simulated_annealing(jobs, initial_temp, cooling_rate, max_time, c_t, num_iterations, seed_value):
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
    shifts = np.arange(-40, 41)  # Shifts range from -20 to 20 inclusive
    shifts = shifts[shifts != 0]  # Remove the zero to avoid no-op moves

    while iterations < num_iterations:
        job_index = np.random.randint(0, len(jobs) - 1)
        job = current_solution[job_index]
        old_start = job['start_time']
        shift = np.random.choice(shifts)
        new_start = old_start + shift

        if 0 <= new_start <= max_time - job['processing_time'] and is_within_constraints(job, new_start):
            new_objective, new_overload, delta_tardiness = evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness)

            if new_objective < current_objective or np.random.random() < np.exp((current_objective - new_objective) / current_temp):
                # Accept the move
                adjust_load(load_at_time, job, new_start, old_start)
                current_solution[job_index]['start_time'] = new_start
                tardiness_values[job_index] = max(0, new_start + job['processing_time'] - job['due_date'])
                total_tardiness = np.sum(tardiness_values)
                total_overload = new_overload
                current_objective = new_objective
                if new_objective < best_objective:
                    best_objective = new_objective

        current_temp *= cooling_rate
        current_temp = max(current_temp, 1)  # Ensuring temperature does not fall below 1
        iterations += 1

    return best_objective  # Return the best objective found after all iteration


def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                    ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                    ('upper_bound', np.int32)])
    jobs = np.array([(1, 72, 73, 511, 435, 40, 80), (2, 20, 26, 440, 391, 40, 80), (3, 28, 75, 463, 460, 40, 80), (4, 66, 53, 138, 130, 40, 80), (5, 59, 77, 58, 78, 40, 80), (6, 71, 62, 40, 31, 40, 80), (7, 45, 36, 58, 22, 40, 80), (8, 50, 75, 175, 152, 40, 80), (9, 46, 75, 54, 62, 40, 80), (10, 71, 46, 458, 401, 40, 80), (11, 80, 32, 332, 324, 40, 80), (12, 58, 74, 145, 156, 40, 80), (13, 80, 44, 126, 112, 40, 80), (14, 72, 47, 161, 136, 40, 80), (15, 57, 60, 306, 315, 40, 80), (16, 79, 64, 360, 327, 40, 80), (17, 55, 30, 70, 55, 40, 80), (18, 38, 41, 79, 110, 40, 80), (19, 78, 61, 140, 120, 40, 80), (20, 31, 75, 327, 271, 40, 80), (21, 43, 61, 465, 382, 40, 80), (22, 67, 68, 298, 295, 40, 80), (23, 68, 80, 358, 370, 40, 80), (24, 44, 20, 39, 3, 40, 80), (25, 34, 73, 10, 9, 40, 80), (26, 40, 80, 61, 68, 40, 80), (27, 51, 54, 178, 130, 40, 80), (28, 35, 67, 303, 346, 40, 80), (29, 38, 34, 112, 111, 40, 80), (30, 62, 50, 255, 174, 40, 80), (31, 29, 54, 10, 51, 40, 80), (32, 78, 50, 78, 54, 40, 80), (33, 68, 48, 472, 418, 40, 80), (34, 36, 52, 103, 108, 40, 80), (35, 75, 78, 259, 197, 40, 80), (36, 29, 58, 138, 90, 40, 80), (37, 31, 43, 154, 188, 40, 80), (38, 21, 48, 401, 352, 40, 80), (39, 30, 39, 366, 408, 40, 80), (40, 24, 79, 417, 404, 40, 80), (41, 27, 54, 337, 322, 40, 80), (42, 48, 41, 169, 117, 40, 80), (43, 57, 50, 48, 65, 40, 80), (44, 60, 56, 23, 32, 40, 80), (45, 77, 53, 9, 3, 40, 80), (46, 55, 61, 178, 203, 40, 80), (47, 66, 40, 331, 229, 40, 80), (48, 50, 68, 208, 144, 40, 80), (49, 62, 33, 294, 254, 40, 80), (50, 27, 29, 189, 168, 40, 80), (51, 63, 43, 413, 425, 40, 80), (52, 39, 61, 288, 226, 40, 80), (53, 58, 22, 337, 287, 40, 80), (54, 76, 65, 255, 222, 40, 80), (55, 70, 37, 79, 25, 40, 80), (56, 63, 70, 447, 433, 40, 80), (57, 34, 49, 380, 421, 40, 80), (58, 26, 46, 450, 470, 40, 80), (59, 42, 24, 450, 387, 40, 80), (60, 77, 49, 88, 41, 40, 80), (61, 58, 72, 140, 128, 40, 80), (62, 37, 71, 283, 239, 40, 80), (63, 46, 44, 497, 452, 40, 80), (64, 79, 71, 306, 198, 40, 80), (65, 47, 24, 283, 293, 40, 80), (66, 57, 20, 208, 169, 40, 80), (67, 29, 30, 27, 72, 40, 80), (68, 68, 38, 6, 18, 40, 80), (69, 34, 29, 174, 162, 40, 80), (70, 32, 47, 427, 452, 40, 80), (71, 28, 79, 21, 14, 40, 80), (72, 28, 37, 85, 49, 40, 80), (73, 75, 51, 33, 33, 40, 80), (74, 69, 28, 13, 12, 40, 80), (75, 24, 27, 127, 172, 40, 80), (76, 39, 54, 196, 236, 40, 80), (77, 50, 35, 92, 30, 40, 80), (78, 39, 36, 40, 25, 40, 80), (79, 21, 55, 435, 408, 40, 80), (80, 75, 77, 417, 388, 40, 80), (81, 61, 78, 447, 435, 40, 80), (82, 63, 76, 465, 461, 40, 80), (83, 65, 47, 497, 409, 40, 80), (84, 40, 24, 303, 258, 40, 80), (85, 53, 80, 463, 486, 40, 80), (86, 69, 35, 258, 263, 40, 80), (87, 21, 24, 92, 99, 40, 80), (88, 61, 58, 329, 308, 40, 80), (89, 29, 43, 447, 438, 40, 80), (90, 49, 33, 12, 6, 40, 80), (91, 80, 58, 240, 139, 40, 80), (92, 55, 67, 61, 0, 40, 80), (93, 22, 57, 39, 39, 40, 80), (94, 45, 26, 392, 424, 40, 80), (95, 36, 41, 120, 95, 40, 80), (96, 52, 40, 210, 214, 40, 80), (97, 64, 77, 424, 396, 40, 80), (98, 74, 27, 126, 107, 40, 80), (99, 53, 70, 58, 17, 40, 80), (100, 50, 39, 312, 317, 40, 80), (101, 77, 64, 13, 13, 40, 80), (102, 45, 20, 13, 19, 40, 80), (103, 31, 51, 61, 48, 40, 80), (104, 39, 49, 201, 174, 40, 80), (105, 47, 64, 332, 279, 40, 80), (106, 55, 62, 219, 221, 40, 80), (107, 23, 24, 169, 176, 40, 80), (108, 38, 72, 161, 117, 40, 80), (109, 43, 31, 23, 45, 40, 80), (110, 30, 26, 294, 273, 40, 80), (111, 41, 39, 332, 345, 40, 80), (112, 77, 20, 497, 496, 40, 80), (113, 38, 67, 43, 78, 40, 80), (114, 59, 58, 161, 135, 40, 80), (115, 57, 47, 235, 244, 40, 80), (116, 25, 34, 294, 233, 40, 80), (117, 79, 63, 219, 111, 40, 80), (118, 59, 31, 205, 110, 40, 80), (119, 65, 23, 176, 168, 40, 80), (120, 46, 26, 14, 27, 40, 80), (121, 72, 72, 184, 93, 40, 80), (122, 78, 61, 296, 193, 40, 80), (123, 75, 77, 10, 8, 40, 80), (124, 64, 59, 366, 276, 40, 80), (125, 22, 72, 347, 311, 40, 80), (126, 70, 60, 43, 52, 40, 80), (127, 64, 63, 103, 72, 40, 80), (128, 34, 28, 332, 339, 40, 80), (129, 65, 79, 64, 16, 40, 80), (130, 26, 38, 275, 212, 40, 80), (131, 61, 48, 118, 42, 40, 80), (132, 70, 39, 261, 185, 40, 80), (133, 46, 47, 249, 275, 40, 80), (134, 54, 66, 329, 291, 40, 80), (135, 32, 57, 465, 472, 40, 80), (136, 68, 41, 43, 6, 40, 80), (137, 21, 77, 201, 156, 40, 80), (138, 21, 45, 23, 55, 40, 80), (139, 42, 45, 329, 291, 40, 80), (140, 29, 26, 149, 154, 40, 80), (141, 42, 52, 294, 247, 40, 80), (142, 44, 26, 107, 60, 40, 80), (143, 23, 63, 214, 202, 40, 80), (144, 27, 49, 489, 535, 40, 80), (145, 41, 57, 417, 446, 40, 80), (146, 54, 64, 465, 403, 40, 80), (147, 61, 21, 6, 12, 40, 80), (148, 54, 35, 48, 0, 40, 80), (149, 80, 68, 440, 327, 40, 80), (150, 52, 78, 178, 104, 40, 80), (151, 80, 48, 501, 480, 40, 80), (152, 47, 75, 149, 106, 40, 80), (153, 51, 41, 392, 395, 40, 80), (154, 75, 42, 261, 188, 40, 80), (155, 47, 66, 124, 113, 40, 80), (156, 64, 61, 435, 413, 40, 80), (157, 74, 29, 378, 317, 40, 80), (158, 56, 54, 358, 358, 40, 80), (159, 66, 40, 289, 201, 40, 80), (160, 29, 46, 78, 99, 40, 80), (161, 50, 37, 208, 229, 40, 80), (162, 29, 57, 154, 174, 40, 80), (163, 29, 47, 413, 434, 40, 80), (164, 80, 21, 48, 22, 40, 80), (165, 44, 29, 288, 276, 40, 80), (166, 61, 50, 235, 201, 40, 80), (167, 74, 22, 392, 298, 40, 80), (168, 72, 57, 289, 231, 40, 80), (169, 68, 35, 112, 47, 40, 80), (170, 42, 45, 380, 307, 40, 80), (171, 59, 61, 204, 139, 40, 80), (172, 34, 41, 424, 375, 40, 80), (173, 56, 56, 334, 247, 40, 80), (174, 57, 24, 118, 139, 40, 80), (175, 52, 52, 501, 461, 40, 80), (176, 72, 76, 205, 147, 40, 80), (177, 43, 22, 306, 334, 40, 80), (178, 34, 36, 294, 285, 40, 80), (179, 54, 48, 255, 233, 40, 80), (180, 75, 74, 283, 267, 40, 80), (181, 67, 38, 249, 203, 40, 80), (182, 73, 28, 130, 44, 40, 80), (183, 79, 43, 372, 341, 40, 80), (184, 20, 33, 126, 86, 40, 80), (185, 79, 53, 288, 218, 40, 80), (186, 21, 61, 307, 274, 40, 80), (187, 79, 67, 61, 37, 40, 80), (188, 37, 21, 346, 355, 40, 80), (189, 24, 79, 306, 287, 40, 80), (190, 69, 76, 10, 21, 40, 80), (191, 36, 22, 130, 153, 40, 80), (192, 47, 80, 358, 375, 40, 80), (193, 62, 29, 79, 39, 40, 80), (194, 32, 58, 204, 153, 40, 80), (195, 36, 39, 390, 325, 40, 80), (196, 26, 36, 120, 172, 40, 80), (197, 38, 30, 520, 482, 40, 80), (198, 38, 26, 28, 5, 40, 80), (199, 65, 68, 288, 232, 40, 80), (200, 80, 59, 127, 122, 40, 80), (201, 72, 62, 347, 336, 40, 80), (202, 37, 46, 169, 94, 40, 80), (203, 80, 59, 500, 492, 40, 80), (204, 76, 23, 161, 115, 40, 80), (205, 42, 58, 205, 125, 40, 80), (206, 28, 63, 331, 381, 40, 80), (207, 70, 22, 33, 29, 40, 80), (208, 35, 57, 179, 183, 40, 80), (209, 57, 33, 81, 42, 40, 80), (210, 79, 23, 511, 442, 40, 80), (211, 67, 24, 401, 298, 40, 80), (212, 55, 53, 489, 456, 40, 80), (213, 59, 20, 503, 471, 40, 80), (214, 78, 27, 174, 108, 40, 80), (215, 24, 59, 205, 162, 40, 80), (216, 43, 25, 201, 183, 40, 80), (217, 41, 48, 27, 43, 40, 80), (218, 39, 70, 210, 160, 40, 80), (219, 30, 21, 330, 341, 40, 80), (220, 52, 61, 307, 234, 40, 80), (221, 80, 80, 303, 284, 40, 80), (222, 70, 36, 413, 404, 40, 80), (223, 30, 26, 184, 126, 40, 80), (224, 60, 29, 383, 330, 40, 80), (225, 71, 68, 219, 159, 40, 80), (226, 36, 38, 511, 455, 40, 80), (227, 64, 30, 174, 72, 40, 80), (228, 68, 77, 133, 71, 40, 80), (229, 20, 51, 296, 247, 40, 80), (230, 44, 59, 438, 457, 40, 80), (231, 35, 38, 112, 92, 40, 80), (232, 21, 32, 124, 78, 40, 80), (233, 45, 26, 360, 309, 40, 80), (234, 79, 54, 219, 171, 40, 80), (235, 79, 33, 40, 41, 40, 80), (236, 63, 67, 6, 0, 40, 80), (237, 50, 22, 126, 145, 40, 80), (238, 36, 23, 331, 301, 40, 80), (239, 45, 34, 294, 301, 40, 80), (240, 61, 77, 380, 334, 40, 80), (241, 21, 62, 174, 206, 40, 80), (242, 36, 24, 440, 484, 40, 80), (243, 41, 27, 39, 35, 40, 80), (244, 24, 55, 183, 179, 40, 80), (245, 31, 46, 145, 164, 40, 80), (246, 44, 60, 13, 11, 40, 80), (247, 51, 71, 120, 35, 40, 80), (248, 40, 45, 337, 373, 40, 80), (249, 35, 65, 511, 444, 40, 80), (250, 76, 31, 330, 257, 40, 80), (251, 74, 78, 9, 5, 40, 80), (252, 27, 74, 360, 360, 40, 80), (253, 46, 59, 447, 454, 40, 80), (254, 80, 80, 288, 227, 40, 80), (255, 57, 52, 390, 349, 40, 80), (256, 31, 63, 136, 101, 40, 80), (257, 41, 23, 261, 188, 40, 80), (258, 78, 52, 487, 375, 40, 80), (259, 66, 55, 306, 303, 40, 80), (260, 77, 63, 255, 231, 40, 80), (261, 55, 62, 213, 183, 40, 80), (262, 52, 64, 306, 275, 40, 80), (263, 35, 54, 176, 162, 40, 80), (264, 32, 65, 240, 258, 40, 80), (265, 61, 36, 517, 468, 40, 80), (266, 43, 30, 322, 259, 40, 80), (267, 74, 30, 40, 35, 40, 80), (268, 79, 60, 472, 414, 40, 80), (269, 44, 71, 176, 125, 40, 80), (270, 75, 22, 58, 56, 40, 80), (271, 38, 29, 39, 20, 40, 80), (272, 35, 36, 346, 280, 40, 80), (273, 29, 74, 347, 392, 40, 80), (274, 56, 35, 3, 22, 40, 80), (275, 35, 67, 363, 325, 40, 80), (276, 72, 65, 450, 387, 40, 80), (277, 45, 28, 161, 193, 40, 80), (278, 27, 50, 503, 510, 40, 80), (279, 61, 80, 210, 200, 40, 80), (280, 63, 62, 92, 46, 40, 80), (281, 76, 50, 327, 231, 40, 80), (282, 34, 59, 438, 411, 40, 80), (283, 67, 71, 259, 254, 40, 80), (284, 38, 71, 107, 39, 40, 80), (285, 26, 76, 154, 206, 40, 80), (286, 52, 48, 219, 170, 40, 80), (287, 41, 25, 465, 459, 40, 80), (288, 30, 33, 334, 382, 40, 80), (289, 64, 45, 9, 23, 40, 80), (290, 33, 26, 213, 165, 40, 80), (291, 47, 43, 383, 315, 40, 80), (292, 74, 77, 322, 275, 40, 80), (293, 40, 68, 184, 215, 40, 80), (294, 35, 46, 330, 354, 40, 80), (295, 57, 21, 435, 445, 40, 80), (296, 79, 20, 472, 377, 40, 80), (297, 25, 22, 37, 82, 40, 80), (298, 39, 24, 255, 240, 40, 80), (299, 21, 32, 154, 110, 40, 80), (300, 71, 44, 237, 146, 40, 80), (301, 31, 78, 366, 404, 40, 80), (302, 38, 58, 472, 480, 40, 80), (303, 53, 37, 332, 276, 40, 80), (304, 44, 22, 497, 434, 40, 80), (305, 55, 49, 288, 226, 40, 80), (306, 47, 49, 327, 261, 40, 80), (307, 26, 67, 230, 213, 40, 80), (308, 78, 21, 149, 150, 40, 80), (309, 77, 80, 273, 267, 40, 80), (310, 50, 33, 330, 307, 40, 80), (311, 32, 28, 392, 397, 40, 80), (312, 56, 24, 183, 188, 40, 80), (313, 47, 54, 330, 262, 40, 80), (314, 21, 38, 189, 136, 40, 80), (315, 66, 50, 154, 90, 40, 80), (316, 27, 43, 106, 71, 40, 80), (317, 80, 38, 450, 360, 40, 80), (318, 59, 37, 273, 178, 40, 80), (319, 43, 24, 10, 31, 40, 80), (320, 49, 62, 424, 419, 40, 80), (321, 74, 20, 296, 220, 40, 80), (322, 47, 68, 138, 86, 40, 80), (323, 32, 78, 404, 428, 40, 80), (324, 46, 38, 140, 160, 40, 80), (325, 79, 58, 487, 392, 40, 80), (326, 37, 39, 503, 485, 40, 80), (327, 76, 31, 204, 173, 40, 80), (328, 33, 27, 61, 83, 40, 80), (329, 30, 44, 332, 331, 40, 80), (330, 23, 23, 58, 59, 40, 80), (331, 76, 50, 306, 208, 40, 80), (332, 43, 21, 435, 453, 40, 80), (333, 31, 35, 39, 76, 40, 80), (334, 39, 69, 258, 260, 40, 80), (335, 49, 63, 3, 1, 40, 80), (336, 76, 56, 88, 62, 40, 80), (337, 78, 27, 53, 3, 40, 80), (338, 31, 42, 472, 428, 40, 80), (339, 26, 62, 427, 480, 40, 80), (340, 54, 29, 520, 497, 40, 80), (341, 64, 67, 465, 476, 40, 80), (342, 52, 49, 183, 200, 40, 80), (343, 61, 74, 258, 194, 40, 80), (344, 45, 30, 214, 241, 40, 80), (345, 40, 52, 201, 169, 40, 80), (346, 51, 21, 54, 47, 40, 80), (347, 28, 45, 92, 80, 40, 80), (348, 38, 59, 266, 228, 40, 80), (349, 67, 79, 413, 401, 40, 80), (350, 24, 34, 120, 158, 40, 80), (351, 78, 43, 401, 283, 40, 80), (352, 26, 54, 296, 245, 40, 80), (353, 37, 68, 179, 183, 40, 80), (354, 76, 48, 40, 43, 40, 80), (355, 49, 41, 261, 229, 40, 80), (356, 24, 34, 337, 283, 40, 80), (357, 58, 21, 81, 96, 40, 80), (358, 24, 70, 360, 315, 40, 80), (359, 62, 20, 43, 30, 40, 80), (360, 58, 48, 458, 429, 40, 80), (361, 61, 72, 133, 73, 40, 80), (362, 64, 39, 401, 363, 40, 80), (363, 33, 80, 138, 155, 40, 80), (364, 62, 29, 366, 281, 40, 80), (365, 45, 45, 190, 168, 40, 80), (366, 63, 75, 88, 30, 40, 80), (367, 66, 37, 435, 388, 40, 80), (368, 40, 73, 21, 9, 40, 80), (369, 50, 31, 425, 356, 40, 80), (370, 68, 70, 138, 140, 40, 80), (371, 29, 52, 71, 89, 40, 80), (372, 58, 79, 118, 46, 40, 80), (373, 70, 64, 367, 320, 40, 80), (374, 62, 78, 237, 159, 40, 80), (375, 75, 54, 136, 90, 40, 80), (376, 40, 46, 337, 367, 40, 80), (377, 67, 27, 235, 132, 40, 80), (378, 46, 35, 169, 84, 40, 80), (379, 77, 68, 210, 190, 40, 80), (380, 46, 50, 103, 96, 40, 80), (381, 44, 25, 366, 350, 40, 80), (382, 47, 53, 303, 230, 40, 80), (383, 20, 63, 288, 304, 40, 80), (384, 27, 67, 503, 486, 40, 80), (385, 79, 29, 138, 63, 40, 80), (386, 32, 43, 480, 505, 40, 80), (387, 71, 30, 487, 423, 40, 80), (388, 43, 56, 294, 248, 40, 80), (389, 32, 66, 273, 252, 40, 80), (390, 29, 37, 372, 330, 40, 80), (391, 28, 29, 3, 32, 40, 80), (392, 36, 51, 401, 394, 40, 80), (393, 35, 29, 64, 58, 40, 80), (394, 39, 43, 298, 251, 40, 80), (395, 61, 28, 258, 270, 40, 80), (396, 20, 46, 201, 240, 40, 80), (397, 72, 20, 208, 183, 40, 80), (398, 61, 52, 383, 325, 40, 80), (399, 70, 22, 40, 35, 40, 80), (400, 62, 52, 266, 217, 40, 80)
    ], dtype=Job)

    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    
    c_t = 2200
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    num_iterations = 4000000
    cooling_rates = [0.9999, 0.99995, 0.99999, 0.999995]
    
    # Step 1: Set the initial temperature to 1000 directly
    initial_temp = len(jobs)*10
    
    # Step 2: Test cooling rates for the fixed initial temperature
    best_overall_objective = float('inf')
    best_config = None

    for cooling_rate in cooling_rates:
        objective_values = []
        durations_list = []

        for i in range(32):
            seed_value = seeds[i % len(seeds)]  # Wrap around seeds if more than 10 runs

            load_at_time = calculate_load_at_time(jobs, max_time)
            total_overload = calculate_total_overload(load_at_time, c_t)
            tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in jobs], dtype=np.int32)
            total_tardiness = np.sum(tardiness_values)
            initial_objective = calculate_objective_value(total_overload, total_tardiness)

            start_time = time.time()

            current_objective = simulated_annealing(
                jobs, initial_temp, cooling_rate, max_time, c_t, num_iterations, seed_value
            )

            duration = time.time() - start_time

            if i > 1:  # Exclude the first two runs for JIT compilation
                objective_values.append(current_objective)
                durations_list.append(duration)

        mean_objective = np.mean(objective_values)
        std_dev_objective = np.std(objective_values)
        min_objective = np.min(objective_values)
        max_objective = np.max(objective_values)
        q1_objective = np.percentile(objective_values, 25)
        q3_objective = np.percentile(objective_values, 75)
        average_duration = np.mean(durations_list)

        print(f"\nSummary for Initial Temp = {initial_temp}, Cooling Rate = {cooling_rate}:")
        print(f"Mean Objective Value: {mean_objective:.2f}")
        print(f"Standard Deviation: {std_dev_objective:.2f}")
        print(f"Minimum Objective Value: {min_objective:.2f}")
        print(f"Maximum Objective Value: {max_objective:.2f}")
        print(f"Q1 Objective (25th percentile): {q1_objective:.2f}")
        print(f"Q3 Objective (75th percentile): {q3_objective:.2f}")
        print(f"Average Duration: {average_duration:.3f} seconds")

        if mean_objective < best_overall_objective:
            best_overall_objective = mean_objective
            best_config = (initial_temp, cooling_rate)

    print(f"\nBest Overall Mean Objective Value: {best_overall_objective}")
    print(f"Best Configuration: Initial Temp = {best_config[0]}, Cooling Rate = {best_config[1]}")

if __name__ == "__main__":
    main()
