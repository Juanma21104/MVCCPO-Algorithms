import numpy as np

# This module contains functions to evaluate individuals and populations to their objectives.

def evaluate(individual, returns, cov_matrix):
    """
    Evaluate the portfolio represented by the individual.

    Parameters:
    - individual: A portfolio represented as a vector of weights.

    Returns:
    - expected_return: Expected return of the portfolio.
    - risk: Risk (variance) of the portfolio.
    """
    expected_return = np.dot(individual, returns) # Expected return is the dot product of weights and expected returns
    risk = np.dot(individual.T, np.dot(cov_matrix, individual)) # Risk = w T * cov_matrix * w
    return expected_return, risk

def precompute_objectives(population, returns, cov_matrix):
    """
    Precompute the returns and risks for all individuals in the population.
    
    Parameters:
    - population: A 2D array representing the population of portfolios.
    - returns: A 1D array representing the expected returns of each asset.
    - cov_matrix: A 2D array representing the covariance matrix of the assets.
    
    Returns:
    - matrix: A 2D array containing the returns and risks of the population.
    """
    N = len(population)
    matrix = np.zeros((2, N))
    for i in range(N):
        matrix[:, i] = evaluate(population[i], returns, cov_matrix)

    return matrix