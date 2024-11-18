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
    all_shifts = np.arange(-10, 11)
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
    jobs = np.array([(1, 12, 10, 14, 20, 10, 20), (2, 12, 12, 112, 95, 10, 20), (3, 12, 17, 221, 217, 10, 20), (4, 9, 8, 203, 214, 10, 20), (5, 11, 6, 11, 17, 10, 20), (6, 9, 7, 129, 118, 10, 20), (7, 6, 15, 85, 87, 10, 20), (8, 14, 14, 227, 230, 10, 20), (9, 20, 16, 14, 12, 10, 20), (10, 10, 8, 15, 25, 10, 20), (11, 10, 19, 79, 72, 10, 20), (12, 19, 12, 79, 73, 10, 20), (13, 5, 10, 114, 127, 10, 20), (14, 12, 16, 32, 12, 10, 20), (15, 7, 15, 21, 23, 10, 20), (16, 5, 18, 79, 82, 10, 20), (17, 15, 13, 125, 117, 10, 20), (18, 13, 14, 135, 134, 10, 20), (19, 19, 10, 118, 92, 10, 20), (20, 13, 11, 135, 134, 10, 20), (21, 13, 15, 174, 178, 10, 20), (22, 9, 9, 174, 168, 10, 20), (23, 11, 17, 192, 175, 10, 20), (24, 18, 19, 11, 3, 10, 20), (25, 13, 15, 114, 105, 10, 20), (26, 19, 5, 32, 9, 10, 20), (27, 7, 19, 79, 63, 10, 20), (28, 20, 15, 135, 114, 10, 20), (29, 14, 19, 128, 107, 10, 20), (30, 6, 16, 34, 19, 10, 20), (31, 13, 13, 221, 217, 10, 20), (32, 14, 17, 192, 187, 10, 20), (33, 11, 7, 34, 23, 10, 20), (34, 5, 5, 125, 123, 10, 20), (35, 19, 15, 32, 26, 10, 20), (36, 15, 6, 66, 58, 10, 20), (37, 20, 7, 11, 10, 10, 20), (38, 6, 8, 66, 67, 10, 20), (39, 6, 9, 90, 92, 10, 20), (40, 14, 13, 196, 182, 10, 20), (41, 17, 17, 128, 120, 10, 20), (42, 17, 6, 112, 102, 10, 20), (43, 5, 20, 11, 14, 10, 20), (44, 8, 9, 196, 186, 10, 20), (45, 7, 9, 118, 114, 10, 20), (46, 13, 8, 32, 21, 10, 20), (47, 19, 16, 54, 31, 10, 20), (48, 18, 5, 14, 1, 10, 20), (49, 7, 8, 34, 41, 10, 20), (50, 8, 13, 79, 86, 10, 20), (51, 5, 10, 118, 116, 10, 20), (52, 14, 16, 112, 94, 10, 20), (53, 11, 19, 128, 110, 10, 20), (54, 20, 17, 118, 94, 10, 20), (55, 16, 16, 227, 223, 10, 20), (56, 7, 12, 135, 142, 10, 20), (57, 20, 9, 227, 205, 10, 20), (58, 18, 13, 230, 218, 10, 20), (59, 8, 15, 227, 212, 10, 20), (60, 11, 17, 85, 86, 10, 20), (61, 20, 14, 196, 193, 10, 20), (62, 19, 12, 230, 206, 10, 20), (63, 14, 10, 118, 101, 10, 20), (64, 5, 15, 46, 33, 10, 20), (65, 20, 17, 15, 9, 10, 20), (66, 20, 17, 11, 8, 10, 20), (67, 6, 18, 15, 20, 10, 20), (68, 19, 14, 85, 82, 10, 20), (69, 6, 16, 174, 182, 10, 20), (70, 6, 13, 118, 118, 10, 20), (71, 8, 10, 11, 2, 10, 20), (72, 14, 7, 183, 175, 10, 20), (73, 9, 11, 54, 52, 10, 20), (74, 19, 11, 125, 101, 10, 20), (75, 13, 11, 114, 95, 10, 20), (76, 16, 10, 114, 93, 10, 20), (77, 8, 20, 118, 123, 10, 20), (78, 12, 8, 54, 58, 10, 20), (79, 20, 9, 227, 219, 10, 20), (80, 9, 19, 129, 115, 10, 20), (81, 7, 10, 118, 116, 10, 20), (82, 13, 11, 129, 113, 10, 20), (83, 11, 19, 54, 37, 10, 20), (84, 5, 5, 15, 16, 10, 20), (85, 7, 20, 114, 115, 10, 20), (86, 11, 14, 66, 55, 10, 20), (87, 19, 11, 178, 162, 10, 20), (88, 16, 6, 11, 10, 10, 20), (89, 14, 10, 32, 14, 10, 20), (90, 14, 7, 192, 189, 10, 20), (91, 12, 10, 178, 174, 10, 20), (92, 17, 5, 227, 207, 10, 20), (93, 11, 15, 54, 56, 10, 20), (94, 16, 16, 196, 199, 10, 20), (95, 14, 11, 15, 10, 10, 20), (96, 14, 11, 135, 111, 10, 20), (97, 16, 10, 178, 156, 10, 20), (98, 10, 9, 54, 55, 10, 20), (99, 7, 11, 79, 90, 10, 20), (100, 9, 20, 54, 52, 10, 20)

], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]
    sequences = [['normal'], ['normal', 'augmented'], ['normal', 'augmented', 'normal'], ['normal', 'augmented', 'normal', 'augmented'], ['normal', 'augmented', 'normal', 'augmented', 'normal']]
    nu = 0.5  # Overload penalty multiplier
    c_t = 100  # Capacity threshold
    max_time = max(job['due_date'] + job['upper_bound'] for job in jobs)
    max_plateau_moves =  8000
    
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