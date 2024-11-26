from numba import njit
import numpy as np
import random
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
    # If in augmented phase, calculate the augmented objective
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
def simplified_random_local_search(jobs, c_t, max_time, seed_value, max_plateau_moves, phase, start_iteration, nu, max_iterations):
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
    all_shifts = np.arange(-10, 11)
    plateau_moves = 0  # Initialize counter for plateau moves
    iteration = start_iteration  # Initialize iteration counter

    # List to store iteration details
    iteration_details = []

    while iteration < start_iteration + max_iterations and plateau_moves < max_plateau_moves:

        # Use the best augmented objective in augmented phase, otherwise the original objective
        min_objective_value = best_augmented_objective_value if phase == 'augmented' else best_original_objective_value

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
            iteration_details.append((iteration, best_original_objective_value, best_augmented_objective_value, plateau_moves, augmented_objective, original_objective))

    # Instead of relying on the last iteration, return best_original_objective_value directly.
    return current_solution, best_original_objective_value, iteration, iteration_details


def local_search_phase_sequence(jobs, c_t, max_time, sequence, max_plateau_moves, nu, seed_value):
    current_solution = jobs.copy()
    cumulative_iteration = 0  # Track cumulative iterations
    all_iterations_trace = []  # To store all iterations and their objective values
    best_objective_overall = np.inf  # Initialize to a large number to track the best

    for phase in sequence:
        # Modify to capture the best original objective value
        current_solution, best_original_objective_value, cumulative_iteration, iteration_details = simplified_random_local_search(
            jobs=current_solution,
            c_t=c_t,
            max_time=max_time,
            seed_value=seed_value,
            max_plateau_moves=max_plateau_moves,
            phase=phase,  
            start_iteration=cumulative_iteration,
            nu=nu,
            max_iterations=10000000  # Adjust as necessary
        )

        all_iterations_trace.extend(iteration_details)

        # Track the best objective value across all phases
        best_objective_overall = min(best_objective_overall, best_original_objective_value)

        if cumulative_iteration >= 10000000:
            break  # Terminate if max iterations reached or constraints violated

    # Return best overall objective and iteration details
    return current_solution, best_objective_overall, cumulative_iteration, all_iterations_trace



def main():
    Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
    jobs = np.array([(1, 15, 16, 3, 0, 10, 20), (2, 19, 16, 117, 91, 10, 20), (3, 6, 5, 15, 9, 10, 20), (4, 17, 15, 121, 95, 10, 20), (5, 19, 13, 121, 94, 10, 20), (6, 16, 14, 25, 28, 10, 20), (7, 5, 7, 47, 62, 10, 20), (8, 18, 14, 7, 9, 10, 20), (9, 17, 14, 31, 18, 10, 20), (10, 15, 16, 76, 62, 10, 20), (11, 20, 11, 7, 4, 10, 20), (12, 17, 17, 29, 2, 10, 20), (13, 10, 9, 117, 125, 10, 20), (14, 15, 20, 121, 102, 10, 20), (15, 9, 14, 90, 88, 10, 20), (16, 5, 8, 104, 116, 10, 20), (17, 11, 17, 50, 33, 10, 20), (18, 17, 6, 39, 42, 10, 20), (19, 20, 13, 7, 2, 10, 20), (20, 9, 5, 55, 36, 10, 20), (21, 12, 10, 55, 60, 10, 20), (22, 7, 7, 62, 70, 10, 20), (23, 17, 20, 47, 33, 10, 20), (24, 10, 20, 34, 26, 10, 20), (25, 5, 18, 48, 52, 10, 20), (26, 13, 5, 34, 33, 10, 20), (27, 7, 17, 28, 14, 10, 20), (28, 6, 9, 49, 57, 10, 20), (29, 15, 12, 76, 75, 10, 20), (30, 9, 16, 48, 46, 10, 20), (31, 14, 20, 117, 122, 10, 20), (32, 12, 15, 28, 16, 10, 20), (33, 6, 17, 121, 131, 10, 20), (34, 9, 13, 15, 23, 10, 20), (35, 10, 19, 48, 57, 10, 20), (36, 12, 7, 15, 17, 10, 20), (37, 10, 20, 90, 96, 10, 20), (38, 11, 6, 121, 108, 10, 20), (39, 15, 6, 49, 31, 10, 20), (40, 17, 16, 55, 42, 10, 20), (41, 16, 7, 34, 18, 10, 20), (42, 7, 16, 31, 23, 10, 20), (43, 11, 5, 121, 103, 10, 20), (44, 14, 13, 124, 108, 10, 20), (45, 20, 9, 90, 84, 10, 20), (46, 14, 10, 15, 20, 10, 20), (47, 5, 7, 55, 61, 10, 20), (48, 19, 18, 15, 10, 10, 20), (49, 15, 11, 28, 33, 10, 20), (50, 8, 14, 3, 13, 10, 20), (51, 20, 15, 34, 29, 10, 20), (52, 11, 9, 49, 43, 10, 20), (53, 6, 10, 29, 13, 10, 20), (54, 12, 19, 34, 32, 10, 20), (55, 7, 13, 121, 134, 10, 20), (56, 18, 15, 49, 48, 10, 20), (57, 16, 20, 26, 19, 10, 20), (58, 13, 15, 124, 130, 10, 20), (59, 10, 14, 76, 74, 10, 20), (60, 12, 11, 121, 100, 10, 20), (61, 5, 9, 26, 37, 10, 20), (62, 11, 14, 76, 56, 10, 20), (63, 20, 7, 3, 0, 10, 20), (64, 19, 5, 31, 25, 10, 20), (65, 6, 13, 76, 87, 10, 20), (66, 8, 5, 49, 34, 10, 20), (67, 8, 6, 76, 58, 10, 20), (68, 5, 20, 55, 51, 10, 20), (69, 10, 18, 62, 63, 10, 20), (70, 19, 16, 15, 1, 10, 20), (71, 14, 19, 47, 46, 10, 20), (72, 14, 10, 34, 11, 10, 20), (73, 8, 9, 90, 81, 10, 20), (74, 20, 8, 117, 101, 10, 20), (75, 20, 8, 26, 26, 10, 20), (76, 13, 6, 28, 21, 10, 20), (77, 15, 8, 55, 60, 10, 20), (78, 11, 11, 15, 22, 10, 20), (79, 17, 10, 121, 111, 10, 20), (80, 7, 15, 124, 135, 10, 20), (81, 10, 12, 104, 104, 10, 20), (82, 6, 12, 130, 139, 10, 20), (83, 18, 8, 130, 129, 10, 20), (84, 10, 18, 7, 1, 10, 20), (85, 13, 9, 90, 77, 10, 20), (86, 8, 19, 25, 7, 10, 20), (87, 7, 11, 29, 14, 10, 20), (88, 19, 11, 28, 27, 10, 20), (89, 8, 10, 26, 33, 10, 20), (90, 13, 18, 48, 27, 10, 20), (91, 16, 10, 39, 21, 10, 20), (92, 18, 9, 55, 42, 10, 20), (93, 18, 16, 62, 63, 10, 20), (94, 14, 13, 90, 86, 10, 20), (95, 11, 6, 15, 1, 10, 20), (96, 11, 18, 121, 123, 10, 20), (97, 10, 10, 55, 46, 10, 20), (98, 16, 10, 49, 30, 10, 20), (99, 5, 10, 130, 139, 10, 20), (100, 11, 20, 25, 31, 10, 20)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]

    sequences = [['normal', 'augmented'], ['normal', 'augmented', 'normal']]
    nu = 1
    c_t = 95
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_plateau_moves = 8000
    
    for sequence in sequences:
        print(f"\nRunning sequence: {sequence}")

        # Store results for each run
        objective_values = []
        iterations_list = []
        durations_list = []

        # Running the search 32 times, calculating summary after 32 runs
        for i in range(32):
            start_time = time.time()

            # Run the local search for the given sequence
            solution, best_objective_overall, final_iteration, iteration_details = local_search_phase_sequence(
                jobs, c_t, max_time, sequence, max_plateau_moves, nu, seed_value=seeds[i]
            )

            duration = time.time() - start_time

            # Skip the first run, which is used for JIT compilation
            if i > 1:
                objective_values.append(best_objective_overall)  # Capture best overall objective value
                iterations_list.append(final_iteration)
                durations_list.append(duration)

        # Calculate summary statistics
        mean_objective = np.mean(objective_values)
        std_dev_objective = np.std(objective_values)
        min_objective = np.min(objective_values)
        max_objective = np.max(objective_values)
        q1_objective = np.percentile(objective_values, 25)
        q3_objective = np.percentile(objective_values, 75)
        average_iterations = np.mean(iterations_list)
        average_duration = np.mean(durations_list)

        # Print summary
        print(f"\nSummary of Objective Values over 30 runs for sequence {sequence}:")
        print(f"Mean Objective Value: {mean_objective:.2f}")
        print(f"Standard Deviation: {std_dev_objective:.2f}")
        print(f"Minimum Objective Value: {min_objective:.2f}")
        print(f"Maximum Objective Value: {max_objective:.2f}")
        print(f"Q1 (25th percentile): {q1_objective:.2f}")
        print(f"Q3 (75th percentile): {q3_objective:.2f}")
        print(f"Average Iterations Needed: {average_iterations:.2f}")
        print(f"Average Duration per Run: {average_duration:.2f} seconds")


if __name__ == "__main__":
    main()