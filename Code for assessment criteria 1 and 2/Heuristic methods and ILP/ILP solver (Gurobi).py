import numpy as np
import gurobipy as gp
from gurobipy import GRB

Job = np.dtype([('id', np.int32), ('processing_time', np.int32), ('weight', np.int32),
                ('due_date', np.int32), ('start_time', np.int32), ('lower_bound', np.int32),
                ('upper_bound', np.int32)])
jobs = np.array([(1,3,1,0,0,6,6), (2,2,1,0,0,6,6), (3,2,1,0,3,6,6), (4,3,1,0,3,6,6)], dtype=Job)

# Number of jobs and time periods
num_jobs = len(jobs)
time_periods = 6
capacity = 1

alpha = 10
beta = 1

# Initial constraint check function
def check_initial_constraints(jobs):
    constraint_violations = []
    for job in jobs:
        job_id, p_j, w_j, d_j, s_j, l_j, u_j = job
        completion_time = s_j + p_j
        if not (d_j - l_j <= completion_time <= d_j + u_j):
            constraint_violations.append(f"Job {job_id} fails with initial completion time {completion_time} not in [{d_j - l_j}, {d_j + u_j}].")
    if constraint_violations:
        print("Initial Constraint Violations:")
        for violation in constraint_violations:
            print(violation)
    else:
        print("All initial constraints satisfied.")

# Check initial constraints
check_initial_constraints(jobs)

# Create a new model
m = gp.Model("job_scheduling")

# Add variables
x = m.addVars(range(1, num_jobs + 1), vtype=GRB.INTEGER, name="start_time")
delta = m.addVars(range(1, num_jobs + 1), range(1, time_periods + 1), vtype=GRB.BINARY, name="delta")
overload = m.addVars(range(1, time_periods + 1), vtype=GRB.INTEGER, name="overload")
tardiness = m.addVars(range(1, num_jobs + 1), vtype=GRB.INTEGER, name="tardiness")

# Rename the timing variables
start_point = m.addVars(range(1, num_jobs + 1), range(1, time_periods + 1), vtype=GRB.BINARY, name="start_point")
end_point = m.addVars(range(1, num_jobs + 1), range(1, time_periods + 1), vtype=GRB.BINARY, name="end_point")
after_end_zero = m.addVars(range(1, num_jobs + 1), range(1, time_periods + 1), vtype=GRB.BINARY, name="after_end_zero")
before_start_zero = m.addVars(range(1, num_jobs + 1), range(1, time_periods + 1), vtype=GRB.BINARY, name="before_start_zero")

# Add constraints
for j, (job_id, p_j, w_j, d_j, s_j, l_j, u_j) in enumerate(jobs, start=1):
    for t in range(1, time_periods + 1):
        # Ensure delta is zero before the start time and after the job finishes
        m.addConstr(start_point[j, t] * (time_periods + 1) >= t - x[j])
        m.addConstr(end_point[j, t] * (time_periods + 1) >= x[j] + p_j - t + 1)
        m.addConstr(after_end_zero[j, t] * (time_periods + 1) >= t - (x[j] + p_j))
        m.addConstr(before_start_zero[j, t] * (time_periods + 1) >= x[j] + 1 - t)

        m.addConstr(delta[j, t] <= 1 - after_end_zero[j, t])  # Force delta to 0 after end
        m.addConstr(delta[j, t] <= 1 - before_start_zero[j, t])  # Force delta to 0 before start
        
        m.addConstr(delta[j, t] >= start_point[j, t] + end_point[j, t] - 1)

    # Constraint to ensure jobs finish within the job-specific time window around due date
    m.addConstr(x[j] + p_j >= d_j - l_j)
    m.addConstr(x[j] + p_j <= d_j + u_j)

    # Calculate tardiness for each job
    m.addConstr(tardiness[j] >= (x[j] + p_j - d_j))
    m.addConstr(tardiness[j] >= 0)

# Constraints for overload
for t in range(1, time_periods + 1):
    total_load_t = gp.quicksum(delta[j, t] * jobs[j-1][2] for j in range(1, num_jobs + 1))
    m.addConstr(overload[t] >= total_load_t - capacity)
    m.addConstr(overload[t] >= 0)

# Objective function: Minimize the weighted sum of total overload and total tardiness
total_overload = gp.quicksum(overload[t] for t in range(1, time_periods + 1))
total_tardiness = gp.quicksum(tardiness[j] for j in range(1, num_jobs + 1))
m.setObjective(alpha * total_overload + beta * total_tardiness, GRB.MINIMIZE)

# Optimize the model
m.optimize()

# Function to check if all 1's in the binary string are connected
def check_connected_ones(delta_values):
    """Check if all 1's in the binary string are connected without being split by '0's."""
    found_one = False  # Flag to indicate we've found at least one '1'
    previous_was_one = False  # Flag to keep track of whether the previous value was '1'

    for value in delta_values:
        if value == 1:
            if not found_one:  # Start of a new sequence of '1's
                found_one = True
            elif not previous_was_one:  # Found a '1' after a '0'
                return False
            previous_was_one = True
        else:
            previous_was_one = False  # Reset this since we're at a '0'

    return found_one  # Only return True if there was at least one '1'

# Collect and check delta values
delta_results = {j: [int(delta[j, t].X) for t in range(1, time_periods + 1)] for j in range(1, num_jobs + 1)}

# Print results and connectivity checks
print("Results and Connectivity Checks:")
for j, deltas in delta_results.items():
    is_connected = check_connected_ones(deltas)
    print(f"Job {j}: Start Time = {x[j].X}, Tardiness = {tardiness[j].X}, Delta: {''.join(map(str, deltas))}, Connected: {is_connected}")

# Print the final objective value
if m.status == GRB.OPTIMAL:
    print(f"Final Objective Value: {m.objVal}")