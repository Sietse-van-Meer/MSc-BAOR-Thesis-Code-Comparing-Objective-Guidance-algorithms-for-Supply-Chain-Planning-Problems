# Comparing Objective-Guidance Algorithms for Supply Chain Planning Problems

This repository contains the code used for my MSc thesis at Tilburg University (MSc Business Analytics & Operations Research, 2024):

The thesis studies the **1-Dimensional Planning Problem (1DPP)**, a single-machine planning problem in which the objective is to minimize a weighted combination of **overload** (overtime capacity usage) and **tardiness** (late completion). We compare performance of **Objective-Guidance Algorithms (O-GAs)** to state-of-the-art metaheuristics such as **Simulated Annealing (SA)** and **Tabu Search (TS)**.

## Repository structure

- **Assessment criteria 1 and 2**  
  Code for experiments on:
  1) mean objective function value  
  2) mean standard deviation of the objective function value
  Also includes the exact solver ILP code (Gurobi).

- **Assessment criteria 4**  
  Code for experiments on mean algorithm complexity.

- **Data Generation Process and dataset files**  
  Dataset generation scripts and the dataset files used in the experiments. You can copy paste the jobs set in other parts of the code to run the code for that dataset.
  
- **Parameter configuration and graphs**

  - **Graphs parameter configuration results**
        This is the code to visualize the parameter configuration results.

  - **Graphs results criterion 1**
        This is the code to visualize the results of assessment criterion 1, mean objective value.

  - **Parameter configuration code**
        This is the code used for configuration of the algorithm parameters.



## Methodology summary

We aim to evaluate algorithms as fairly as possible by:
- using **21 datasets**, ranging from small instances (100 jobs) to large instances (800 jobs),
- standardizing implementations to reduce the influence of coding choices,
- tuning parameters on a subset of datasets and evaluating a single configuration across all datasets.

Algorithms are compared on four criteria:
1. **Mean objective function value**  
2. **Mean standard deviation of the objective function value**  
3. **Parameter sensitivity**  
4. **Mean algorithm complexity**

## Reproducibility

- Runs are controlled with **fixed random seeds** for consistency.
- Each configuration is run **32 times** per dataset.
- Reported averages use the **last 30 runs**, excluding the first 2 runs due to **Numba JIT warm-up**.

## Requirements

The codebase is written in Python and uses performance optimizations such as **Numba**.

Typical requirements include:
- Python 3.10+
- numpy
- numba
- pandas
- matplotlib

If you have a `requirements.txt`, install with all Python packages with:

```bash
pip install -r requirements.txt
```
