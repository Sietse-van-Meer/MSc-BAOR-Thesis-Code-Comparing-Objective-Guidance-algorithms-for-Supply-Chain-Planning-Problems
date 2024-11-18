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
def calculate_earliness_for_job(job):
    start_time = job['start_time']
    due_date = job['due_date']
    duration = job['processing_time']
    finish_time = start_time + duration
    earliness = max(0, due_date - finish_time)  # Earliness is calculated if job finishes before the due date
    return earliness


@njit(cache=True)
def calculate_augmented_objective_value(jobs, load_at_time, total_tardiness, c_t, nu, earliness_values, delta_earliness, job_index):
    # Calculate the total overload
    overload = np.maximum(0, load_at_time - c_t)
    total_overload = np.sum(overload)
    
    # Calculate the original objective value (based on total overload and total tardiness)
    original_objective_value = calculate_objective_value(total_overload, total_tardiness)
    
    # Update the earliness value for the moved job
    updated_earliness_values = earliness_values.copy()
    updated_earliness_values[job_index] += delta_earliness

    # Square each job's earliness individually and sum them up
    squared_earliness_sum = np.sum(updated_earliness_values**2)
    
    # If squared earliness sum is positive, apply the penalty
    earliness_penalty = nu * squared_earliness_sum if squared_earliness_sum > 0 else 0
    
    # Final augmented objective value: original objective minus the squared earliness penalty
    augmented_objective_value = original_objective_value - earliness_penalty
    
    return augmented_objective_value

@njit(cache=True)
def evaluate_move(load_at_time, job, new_start, old_start, c_t, total_tardiness, nu, phase, earliness_values, job_index):
    duration = job['processing_time']
    weight = job['weight']
    max_time = len(load_at_time) - 1
    
    # Calculate original earliness and tardiness before the move
    original_earliness = earliness_values[job_index]
    original_tardiness = max(0, old_start + duration - job['due_date'])
    
    # Temporarily adjust load for the new start
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] += weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] -= weight

    # Calculate the new overload and new tardiness
    new_overload = calculate_total_overload(load_at_time, c_t)
    new_tardiness = max(0, new_start + duration - job['due_date'])
    delta_tardiness = new_tardiness - original_tardiness

    # Temporarily update the job's start time for earliness calculation
    job['start_time'] = new_start
    new_earliness = calculate_earliness_for_job(job)
    delta_earliness = new_earliness - original_earliness

    # Restore the job's start time
    job['start_time'] = old_start

    # Calculate original objective (based on total overload and updated total tardiness)
    original_objective = calculate_objective_value(new_overload, total_tardiness + delta_tardiness)

    # Augmented objective calculation if in the augmented phase
    if phase == 'augmented':
        augmented_objective = calculate_augmented_objective_value(
            jobs=None,  # Not needed since we're passing specific parameters
            load_at_time=load_at_time,
            total_tardiness=total_tardiness + delta_tardiness,
            c_t=c_t,
            nu=nu,
            earliness_values=earliness_values,
            delta_earliness=delta_earliness,
            job_index=job_index
        )
    else:
        augmented_objective = original_objective

    # Revert the temporary load adjustment
    for t in range(new_start, min(new_start + duration, max_time + 1)):
        load_at_time[t] -= weight
    for t in range(old_start, min(old_start + duration, max_time + 1)):
        load_at_time[t] += weight

    return original_objective, augmented_objective, new_overload, delta_tardiness, delta_earliness

@njit(cache=True)
def simplified_random_local_search(jobs, c_t, max_time, seed_value, max_plateau_moves, phase, start_iteration, nu, max_iterations=300000):
    np.random.seed(seed_value)
    current_solution = jobs.copy()
    load_at_time = calculate_load_at_time(current_solution, max_time)
    tardiness_values = np.array([max(0, job['start_time'] + job['processing_time'] - job['due_date']) for job in current_solution], dtype=np.int32)
    earliness_values = np.array([calculate_earliness_for_job(job) for job in current_solution], dtype=np.int32)

    total_tardiness = np.sum(tardiness_values)
    total_squared_earliness = np.sum(earliness_values**2)
    total_overload = calculate_total_overload(load_at_time, c_t)

    # Track original and augmented objective values
    current_original_objective = calculate_objective_value(total_overload, total_tardiness)
    best_original_objective_value = current_original_objective  # Track the best found original objective
    best_augmented_objective_value = calculate_augmented_objective_value(
        jobs=None,
        load_at_time=load_at_time,
        total_tardiness=total_tardiness,
        c_t=c_t,
        nu=nu,
        earliness_values=earliness_values,
        delta_earliness=0,  # Initial delta earliness is 0
        job_index=0  # Arbitrary as no job has moved yet
    ) if phase == 'augmented' else np.nan

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
            # Evaluate the move, which calculates both the new objective values and the changes in tardiness and earliness
            original_objective, augmented_objective, new_overload, delta_tardiness, delta_earliness = evaluate_move(
                load_at_time, job, new_start, old_start, c_t, total_tardiness, nu, phase, earliness_values, job_index
            )

            # Update the current original objective after the move
            current_original_objective = calculate_objective_value(new_overload, total_tardiness + delta_tardiness)

            if phase == 'augmented':
                # For augmented phase: evaluate using augmented objective, but track plateau based on original objective
                if augmented_objective <= best_augmented_objective_value:
                    # Accept the move if the augmented objective improves or stays equal
                    best_augmented_objective_value = augmented_objective
                    
                    # Plateau moves are tracked based on the original objective
                    if current_original_objective < best_original_objective_value:
                        plateau_moves = 0  # Improvement in the original objective, reset plateau moves
                        best_original_objective_value = current_original_objective  # Update the best original objective
                    else:
                        plateau_moves += 1  # No improvement or worsening in the original objective
                    
                    # Make the move
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    earliness_values[job_index] += delta_earliness
                    total_tardiness += delta_tardiness
                    total_squared_earliness += (delta_earliness**2)
                    total_overload = new_overload
                    iteration += 1
                else:
                    # Reject the move if augmented objective worsens
                    plateau_moves += 1
                    iteration += 1
            else:
                # Normal phase: consider only the original objective for move acceptance and plateau tracking
                if original_objective < best_original_objective_value:
                    # Accept the move if the original objective improves (goes down)
                    best_original_objective_value = original_objective
                    plateau_moves = 0  # Reset plateau moves on improvement
                    
                    # Make the move
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    earliness_values[job_index] += delta_earliness
                    total_tardiness += delta_tardiness
                    total_squared_earliness += (delta_earliness**2)
                    total_overload = new_overload
                    iteration += 1
                elif original_objective == best_original_objective_value:
                    # If the original objective stays the same, increment plateau moves
                    plateau_moves += 1

                    # Make the move (since it's an equal objective value move)
                    adjust_load(load_at_time, job, new_start, old_start)
                    current_solution[job_index]['start_time'] = new_start
                    tardiness_values[job_index] += delta_tardiness
                    earliness_values[job_index] += delta_earliness
                    total_tardiness += delta_tardiness
                    total_squared_earliness += (delta_earliness**2)
                    total_overload = new_overload
                    iteration += 1
                else:
                    # If the original objective worsens, increment plateau moves and reject the move
                    plateau_moves += 1
                    iteration += 1

            # Store iteration details including evaluated values, best objectives, and plateau moves
            iteration_details.append((
                iteration, 
                best_original_objective_value, 
                best_augmented_objective_value, 
                plateau_moves, 
                augmented_objective, 
                original_objective
            ))

    # Instead of relying on the last iteration, return the best_original_objective_value directly.
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
    jobs = np.array([(1, 9, 13, 206, 195, 10, 20), (2, 19, 10, 82, 79, 10, 20), (3, 10, 20, 82, 71, 10, 20), (4, 15, 20, 156, 159, 10, 20), (5, 10, 15, 167, 160, 10, 20), (6, 8, 6, 80, 65, 10, 20), (7, 11, 13, 57, 43, 10, 20), (8, 17, 16, 175, 159, 10, 20), (9, 15, 7, 104, 107, 10, 20), (10, 14, 9, 156, 140, 10, 20), (11, 8, 12, 70, 57, 10, 20), (12, 5, 16, 190, 203, 10, 20), (13, 10, 10, 190, 186, 10, 20), (14, 15, 5, 167, 152, 10, 20), (15, 19, 7, 229, 218, 10, 20), (16, 18, 16, 65, 67, 10, 20), (17, 8, 9, 112, 114, 10, 20), (18, 11, 16, 136, 123, 10, 20), (19, 9, 8, 181, 187, 10, 20), (20, 9, 6, 57, 62, 10, 20), (21, 13, 9, 79, 76, 10, 20), (22, 5, 12, 56, 49, 10, 20), (23, 13, 9, 80, 81, 10, 20), (24, 11, 16, 175, 155, 10, 20), (25, 17, 19, 80, 63, 10, 20), (26, 20, 18, 201, 196, 10, 20), (27, 5, 8, 152, 161, 10, 20), (28, 11, 16, 175, 159, 10, 20), (29, 12, 8, 161, 168, 10, 20), (30, 9, 7, 65, 65, 10, 20), (31, 19, 16, 57, 58, 10, 20), (32, 9, 17, 224, 233, 10, 20), (33, 10, 16, 224, 232, 10, 20), (34, 10, 6, 136, 121, 10, 20), (35, 5, 13, 80, 67, 10, 20), (36, 16, 19, 181, 181, 10, 20), (37, 15, 13, 156, 148, 10, 20), (38, 11, 19, 65, 51, 10, 20), (39, 15, 19, 201, 202, 10, 20), (40, 12, 8, 229, 212, 10, 20), (41, 8, 19, 161, 158, 10, 20), (42, 9, 10, 190, 192, 10, 20), (43, 13, 15, 161, 165, 10, 20), (44, 9, 9, 190, 185, 10, 20), (45, 16, 18, 224, 205, 10, 20), (46, 11, 7, 230, 234, 10, 20), (47, 5, 20, 136, 139, 10, 20), (48, 14, 17, 224, 215, 10, 20), (49, 18, 14, 190, 173, 10, 20), (50, 14, 15, 70, 56, 10, 20), (51, 11, 5, 156, 162, 10, 20), (52, 13, 16, 104, 96, 10, 20), (53, 17, 6, 181, 168, 10, 20), (54, 20, 8, 175, 165, 10, 20), (55, 13, 8, 82, 81, 10, 20), (56, 10, 9, 156, 150, 10, 20), (57, 16, 15, 152, 130, 10, 20), (58, 20, 13, 206, 179, 10, 20), (59, 9, 8, 112, 114, 10, 20), (60, 11, 19, 201, 189, 10, 20), (61, 14, 18, 175, 164, 10, 20), (62, 6, 9, 80, 65, 10, 20), (63, 10, 7, 161, 163, 10, 20), (64, 19, 13, 136, 116, 10, 20), (65, 20, 14, 206, 203, 10, 20), (66, 16, 14, 57, 38, 10, 20), (67, 9, 15, 224, 225, 10, 20), (68, 20, 13, 175, 167, 10, 20), (69, 14, 7, 3, 2, 10, 20), (70, 16, 10, 82, 57, 10, 20), (71, 10, 13, 152, 145, 10, 20), (72, 17, 20, 181, 163, 10, 20), (73, 14, 8, 175, 162, 10, 20), (74, 7, 8, 167, 175, 10, 20), (75, 12, 11, 104, 90, 10, 20), (76, 9, 18, 175, 168, 10, 20), (77, 14, 6, 230, 209, 10, 20), (78, 10, 5, 181, 165, 10, 20), (79, 11, 20, 65, 62, 10, 20), (80, 10, 14, 152, 135, 10, 20), (81, 10, 12, 229, 227, 10, 20), (82, 14, 9, 201, 193, 10, 20), (83, 18, 8, 201, 192, 10, 20), (84, 17, 17, 181, 159, 10, 20), (85, 10, 8, 229, 238, 10, 20), (86, 7, 19, 57, 41, 10, 20), (87, 11, 16, 112, 114, 10, 20), (88, 10, 16, 136, 127, 10, 20), (89, 19, 6, 79, 63, 10, 20), (90, 6, 11, 161, 170, 10, 20), (91, 8, 15, 167, 161, 10, 20), (92, 19, 12, 56, 47, 10, 20), (93, 8, 12, 70, 73, 10, 20), (94, 17, 9, 112, 112, 10, 20), (95, 16, 17, 161, 165, 10, 20), (96, 17, 9, 224, 210, 10, 20), (97, 5, 7, 224, 224, 10, 20), (98, 9, 16, 181, 179, 10, 20), (99, 17, 9, 57, 38, 10, 20), (100, 12, 20, 57, 45, 10, 20)
], dtype=Job)
    
    seeds = [1415, 9265, 3589, 7932, 3846, 2643, 3832, 7950, 2884, 1971, 6939, 9375, 1058, 2097, 4944, 5923, 781, 6406, 2862, 899, 8628, 348, 2534, 2117, 679, 8214, 8086, 5132, 8230, 6647, 938, 4460]

    sequences = [['normal', 'augmented'], ['normal', 'augmented', 'normal']]
    nu = 1
    c_t = 80
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
        print(f"Average Duration per Run: {average_duration:.3f} seconds")


if __name__ == "__main__":
    main()