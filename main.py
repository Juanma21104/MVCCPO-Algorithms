from algorithms.nsga2 import NSGA2
from algorithms.nsga2_pro import NSGA2Pro
from algorithms.data_loader import load_dataset
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load dataset and run the algorithm
    returns, cov_matrix = load_dataset('data/port1.txt')
    # print(f"Expected Returns: {returns}")
    # print(f"Covariance Matrix:\n{cov_matrix}")

    num_assets = len(returns)
    population_size = 250 # N_pop
    generations = 400 # 
    cardinality = 10
    mutation_rate = 0.9

    N_arc = 250 # Archive population size (A0), is used to store the best solutions found so far
    N_pop = 250 # Usual population size (B0), is used to store the current population
    #
 
    # Run NSGA-II algorithm
    nsga2 = NSGA2(population_size, generations, num_assets, returns, cov_matrix, cardinality, mutation_rate)
    nsga2.evolve()
    nsga2.plot_pareto_front()


    """nsga2 = NSGA2Pro(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    nsga2.evolve()
    nsga2.plot_pareto_front()"""