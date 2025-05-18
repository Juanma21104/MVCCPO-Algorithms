from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.data_loader import load_dataset
from algorithms.utils.visualization import plot_pareto_front as print_pareto_front

if __name__ == "__main__":

    # Load dataset and run the algorithm
    returns, cov_matrix = load_dataset('data/paper_data/port5.txt')
    #print(f"Expected Returns: {returns}")
    #print(f"Covariance Matrix:\n{cov_matrix}")

    num_assets = len(returns)
    population_size = 250 # N_pop
    generations = 400 # 
    cardinality = 10
    mutation_rate = 0.9

    N_arc = 250 # Archive population size (A0), is used to store the best solutions found so far
    N_pop = 250 # Usual population size (B0), is used to store the current population

    tournament_size = 10
    niche_radius = 0.7
 
    # Run NSGA-II 
    """nsga2 = NSGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    final_population = nsga2.evolve()
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "NSGA-II")"""


    # Run SPEA-II algorithm
    """spea2 = SPEA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    final_population = spea2.evolve()
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "SPEA2")"""


    # Run NPGA2 algorithm    
    """npga2 = NPGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "NPGA2")"""


    # Run PESA algorithm    
    pesa = PESA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate - 0.1, generations)
    final_population = pesa.evolve()
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "PESA")


    # Run E-MOEA algorithm    
    """e = 0.00458 * 1.1
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate - 0.1, generations, e)
    final_population = e_moea.evolve()
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "e-MOEA")"""
    



    








    
            