from algorithms import nsga2_pro
from algorithms.nsga2 import NSGA2
from algorithms.nsga2_pro import NSGA2Pro
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.soea import SOEA
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.data_loader import load_dataset
import numpy as np
import algorithms.utils as utils

if __name__ == "__main__":

    # Load dataset and run the algorithm
    returns, cov_matrix = load_dataset('data/port5.txt')
    #print(f"Expected Returns: {returns}")
    #print(f"Covariance Matrix:\n{cov_matrix}")



    num_assets = len(returns)
    population_size = 250 # N_pop
    generations = 400 # 
    cardinality = 10
    mutation_rate = 0.9

    N_arc = 250 # Archive population size (A0), is used to store the best solutions found so far
    N_pop = 250 # Usual population size (B0), is used to store the current population

    tournament_size = 8
    niche_radius = 0.7

    trade_off_coeff = 0.5


    
 
    # Run NSGA-II algorithm
    """nsga2 = NSGA2(population_size, generations, num_assets, returns, cov_matrix, cardinality, mutation_rate)
    nsga2.evolve()
    nsga2.plot_pareto_front()"""


    """nsga2 = NSGA2Pro(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    nsga2.evolve()
    nsga2.plot_pareto_front()"""

    """spea2 = SPEA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    spea2.evolve()
    spea2.plot_pareto_front()"""

    
    """npga2 = NPGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, tournament_size, niche_radius)
    npga2.evolve()
    npga2.plot_pareto_front()"""

    
    """pesa = PESA(N_pop, N_arc, num_assets, returns, cov_matrix, cardinality, mutation_rate - 0.1, generations)
    pesa.evolve()
    pesa.plot_pareto_front()"""


    e = 0.00438
    e_moea = E_MOEA(N_pop, None, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, e)
    e_moea.evolve()
    e_moea.plot_pareto_front()
    


    """soea = SOEA(100, 1, num_assets, returns, cov_matrix, cardinality, mutation_rate, 100000, trade_off_coeff)
    soea.evolve()
    print("Best individual:", soea.get_best_individual())
    ret, risk = utils.evaluate(soea.get_best_individual(), returns, cov_matrix)
    print("Returns and risks:", ret, risk)"""



    








    
            