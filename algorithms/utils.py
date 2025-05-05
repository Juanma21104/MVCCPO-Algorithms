import numpy as np
import random

def initialize_population(N_pop, num_assets, cardinality):
    """
    Initialize the population with random portfolios.


    Returns:
    - population A: A 2D empty array
    - population B: A 2D array representing the initial population of portfolios.
    """
    population_A = np.empty((0, num_assets))
    population_B = []
    for _ in range(N_pop):
        individual = np.zeros(num_assets)
        selected_assets = random.sample(range(num_assets), cardinality) # Select random assets to include in the portfolio
        for asset in selected_assets:
            value = random .uniform(0, 1)
            while value == 0:
                value = random.uniform(0, 1)
            individual[asset] = value
        individual /= individual.sum()
        #individual = projection(individual)
        population_B.append(individual)
        
    return population_A, np.array(population_B)

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
        ret, risk = evaluate(population[i], returns, cov_matrix)
        matrix[0][i] = ret
        matrix[1][i] = risk
    return matrix

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

def dominates(matrix_ret_risks, ind1, ind2):
    """
    Check if individual 1 dominates individual 2.
    
    Parameters:
    - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        - ind1: First individual (portfolio).
        - ind2: Second individual (portfolio).

    Returns:
    - True if ind1 dominates ind2, False otherwise.
    """
    return1 = matrix_ret_risks[0][ind1]
    risk1 = matrix_ret_risks[1][ind1]
    return2 = matrix_ret_risks[0][ind2]
    risk2 = matrix_ret_risks[1][ind2]       
    return (return1 >= return2 and risk1 <= risk2) and (return1 > return2 or risk1 < risk2)

def binary_tournament(population, fitness):
    """
    Perform binary tournament selection.
    
    Parameters:
    - population: The current population.
    - fitness: A 1D array representing the fitness of each individual in the population.
    
    Returns:
    - selected: The selected individual.
    """
    ind1, ind2 = random.sample(range(len(population)), 2)
    return population[ind1] if fitness[ind1] < fitness[ind2] else population[ind2]

def crossover(parent1, parent2, num_assets, cardinality):
    """
    Perform crossover between two parents to create one child.
    
    Parameters:
    - parent1: First parent (portfolio).
    - parent2: Second parent (portfolio).
    
    Returns:
    - child: Child (portfolio) created from the parents.

    """

    child = np.zeros(num_assets)
    parent1_indexes = np.where(parent1 > 0)[0]
    parent2_indexes = np.where(parent2 > 0)[0]

    equal_indexes = np.intersect1d(parent1_indexes, parent2_indexes)

    for index in equal_indexes:
        if random.random() > 0.5:
            child[index] = parent1[index]
        else:
            child[index] = parent2[index]
    
    parent1_indexes = np.setdiff1d(parent1_indexes, equal_indexes)
    parent2_indexes = np.setdiff1d(parent2_indexes, equal_indexes)

    for i in range(cardinality - len(equal_indexes)):
        if random.random() > 0.5:
            child[parent1_indexes[i]] = parent1[parent1_indexes[i]]
        else:
            child[parent2_indexes[i]] = parent2[parent2_indexes[i]]

    child = child / child.sum()
    #child = utils.projection(child)
    return child
    
def mutation(individual, mutation_rate):
    """
    Perform mutation on an individual.
    
    Parameters:
    - individual: Individual (portfolio) to mutate.
    
    Returns:
    - mutated_individual: Mutated individual (portfolio).
    
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] *= random.uniform(0.75, 1.25) # Mutation 25%
    individual = individual / individual.sum()
    #individual = utils.projection(individual) # Normalize the mutated individual to sum to 1
    return individual





def projection(y):
    """
    Ortogonal projection of a vector y onto the set Ω1 := {x ∈ ℝ^S : 1ᵀx = 1}
    """
    y = np.array(y)
    y_indexes = np.where(y > 0)[0]
    y2 = y[y_indexes] # Select positive values

    S = len(y2)
    ones = np.ones(S)
    z = y2 - ((ones.T @ y2 - 1) / S) * ones

    print(len(np.where(z > 0)[0]))

    print(z)
    print(y_indexes)
    z2 = np.zeros(len(y))
    z2[y_indexes] = z

    print(len(np.where(z2 > 0)[0]))

    return z2
    

def projection_simplex(y):
    """
    Proyecta el vector y sobre la simplex de probabilidad:
        Ω2 := {x ∈ ℝ^S : x ≥ 0, sum(x) = 1}
    """
    y = np.asarray(y)
    # Ordenamos y de mayor a menor
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u + (1 - cssv) / (np.arange(1, len(y)+1)) > 0)[0][-1]
    theta = (1 - cssv[rho]) / (rho + 1)
    z = np.maximum(y + theta, 0)
    return z

