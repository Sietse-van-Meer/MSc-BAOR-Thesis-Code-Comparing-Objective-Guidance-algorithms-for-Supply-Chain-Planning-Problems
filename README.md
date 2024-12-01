# README: MSc Thesis - Comparing Objective-Guidance Algorithms for Supply Chain Planning Problems

This repository contains all the input data and code required to reproduce the results and facilitate future research based on the MSc thesis. The contents are organized into four main folders, described below.

---

## Contents

### 1. Data Generating Process Code and Datasets
- **Purpose**: Contains the code and scripts used to generate synthetic datasets 1–20 for the study.
- **Content**:
  - Code to generate (your own) datasets.
  - Pre-generated datasets used in the thesis.
- **Instructions**:
  1. One can copy the data from the dataset one wants to use, which is in form: `(j, p_j, w_j, d_j, x_j, l_j, u_j)` to the experiment they want to try or replicate. Note datasets may have different capacity constraints and neighborhood sizes.

### 2. Associated Tests (Chapter 5)
- **Purpose**: Includes associated tests code from Chapter 5 of the thesis.
- **Content**: Code and configurations used for experiments.
- **Instructions**: Run the scripts as described in the code, and change parameters such that they fit the experiment (e.g., for dataset 1, use neighborhood `(-10, 10)` and `c_t = 80`).

### 3. Criterion 1 and 2: Main Objective Value and Standard Deviation Tests
- **Purpose**: Contains code to generate the main results used to evaluate algorithms based on:
  - Criterion 1: Main Objective Value.
  - Criterion 2: Standard Deviation.
- **Content**:
  - Final algorithm configurations for the tests, implemented to assess mean objective value and their mean standard deviation of the algorithms.
- **Instructions**:
  1. Load datasets as described.
  2. Tweak neighborhood/parameters.
  3. Run scripts to compute results for each algorithm.

### 4. Criterion 4: Algorithm Complexity
- **Purpose**: Contains the code for evaluating algorithm complexity, which evaluates the duration of running 10 million iterations.
- **Content**:
  - Scripts for complexity tests with a single limit of 10 million iterations.
- **Instructions**:
  1. Set the dataset input, machine capacity, and neighborhood correctly. 
  2. Execute the script.

---

## General Instructions for Reproducing Results

To reproduce results for any experiment, follow these steps:
1. **Load Data**: Import `j, p_j, w_j, d_j, x_j, l_j, u_j` from the dataset sheet corresponding to the experiment.
2. **Neighborhood Check**: Verify the neighborhood range settings (e.g., `(-20, 21)`).
3. **Plateau Moves**:
   - **Local Search (LS)** and **Simulated Annealing (SA)**: Plateau moves do not require special handling.
   - **Tabu Search (TS)**: Plateau moves are handled within the function, not in the main script.
4. **Set Maximum Iterations**: Define the maximum number of iterations for the algorithm in the main script.

---

## Generating Graphs

To generate the graphs shown in the thesis:
- Use the scripts in the **Criterion 1 and 2** folder.
- Calculate the mean objective value for each configuration.
- Visualize results using the provided graphing utilities.

---

## Helper Functions

### `calculate_load_at_time()`
- **Purpose**: Adds weight to the periods job by job.

### `adjust_load()`
- **Purpose**: Adjusts the weight by subtracting it from the time periods where the job was active and adding it to the periods where the job becomes active.
- **Optimization Note**: This can be made more efficient by keeping the unchanged periods indexed, though this increases complexity.

### `calculate_total_overload()`
- **Purpose**: Calculates the total overload across all time periods.

### `calculate_objective_value()`
- **Purpose**: Calculates the objective function. Default values for `alpha` and `beta` are `10` and `1`.

### `is_within_constraints()`
- **Purpose**: Ensures the job's finishing time stays within bounds (`d_j - l_j` to `d_j + u_j`).

### `evaluate_move()`
- **Purpose**: Temporarily adjusts the weight (similar to `adjust_load()`), calculates the new overload, delta tardiness, and objective function value, and reverts the adjustment. 
- **Note**: This ensures moves are always tested, reverted, and implemented only if valid.

---

## Algorithm-Specific Notes

### RFILS
- **Settings**:
  - Define shifts (neighborhood) in `simplified_random_local_search`.
  - Set `max_iterations` in `main()`.
  - **Equal Moves**: Stores up to 10 equal moves and selects one if no improvement is found.

### Omega-w
- **Settings**:
  - Define shifts (neighborhood) in `simplified_random_local_search`.
  - Set `nu`, sequence, max plateau moves, and `max_iterations` in `main()`.
  - **Phases**: `normal` for omega phase, `w` for augmented phases.
  - Phases orchestrated via `local_search_phase_sequence`.

### GLS-Q
- **Settings**:
  - Define shifts (neighborhood) and reset penalties when `np.sum(penalties) >= 4` in `simplified_random_local_search`.
  - Set `nu` (lambda), plateau move limit, and `max_iterations` in `main()`.
  - Uses `calculate_quarterly_load` to compute aggregated quarterly loads for penalization.

### Tabu Search (TS)
- **Settings**:
  - Define shifts (neighborhood) and plateau move limits in `simplified_random_local_search`.
  - Set diversification moves, tenure parameters, and `max_iterations` in `main()`.

### Simulated Annealing (SA)
- **Settings**:
  - Define shifts (neighborhood) in `simulated_annealing`.
  - Set initial temperature, cooling rate, and `max_iterations` in `main()`.
  - Cooling rule defined via `get_cooling_rate`.

### LS+SA
- **Settings**:
  - Define shifts (neighborhood) in `hybrid_local_search_simulated_annealing`.
  - Set initial temperature, cooling rate, max plateau moves, and `max_iterations` in `main()`.

### $\omega-w$ + SA
- **Settings**:
  - Define shifts (neighborhood) in both `simplified_random_local_search` and `simulated_annealing`.
  - Set initial temperature, cooling rate, `nu`, sequence, and max plateau moves in `main()`.
  - Orchestrated via `local_search_phase_sequence`.

### ILP
- **Settings**:
  - Set `alpha` and `beta` to reflect the balance between tardiness and overload.
  - **Warning**: The ILP function flags infeasible problem instances.

---

## Notes
- This code is licensed for academic and research purposes.
- Contributions and extensions to the codebase are welcome for future research.

For questions or further details, refer to the thesis document or contact the repository maintainer.
