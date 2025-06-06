from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.utils.data_loader import load_dataset, load_real_data, save_data_txt
from algorithms.utils.visualization import plot_pareto_front as print_pareto_front
from algorithms.utils.performance import calculate_performance

if __name__ == "__main__":

    # Load dataset and run the algorithm
    returns, cov_matrix = load_dataset('data/synthetic_data/Japanese_Nikkei_225.txt')
    #returns, cov_matrix = load_dataset('data/real_data/processed_data/North_America_25_Portfolios_ME_INV_Daily.txt')
    #print(f"Covariance Matrix:\n{cov_matrix}")

    num_assets = len(returns)
    population_size = 250 # N_pop
    generations = 400 # 
    cardinality = 10
    crossover_rate = 0.9
    mutation_rate = 1 / num_assets

    N_arc = 250 # Archive population size (A0), is used to store the best solutions found so far
    N_pop = 250 # Usual population size (B0), is used to store the current population

    tournament_size = 10
    niche_radius = 0.7

    grid_divisions = 10 # Number of divisions in the grid for PESA algorithm
 
    # Run NSGA-II 
    """nsga2 = NSGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = nsga2.evolve()
    indicators = calculate_performance(final_population, returns, cov_matrix)
    print(f"Hypervolume: {indicators['hypervolume']:.3f}, Sharpe Ratio: {indicators['sharpe_ratio']:.3f}")
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "NSGA-II")"""


    # Run SPEA-II algorithm
    """spea2 = SPEA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations)
    final_population = spea2.evolve()
    indicators = calculate_performance(final_population, returns, cov_matrix)
    print(f"Hypervolume: {indicators['hypervolume']:.3f}, Sharpe Ratio: {indicators['sharpe_ratio']:.3f}")
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "SPEA2")"""


    # Run NPGA2 algorithm    
    """npga2 = NPGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate, mutation_rate, generations, tournament_size, niche_radius)
    final_population = npga2.evolve()
    indicators = calculate_performance(final_population, returns, cov_matrix)
    print(f"Hypervolume: {indicators['hypervolume']:.3f}, Sharpe Ratio: {indicators['sharpe_ratio']:.3f}")
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "NPGA2")"""


    # Run PESA algorithm    
    """pesa = PESA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, grid_divisions)
    final_population = pesa.evolve()
    indicators = calculate_performance(final_population, returns, cov_matrix)
    print(f"Hypervolume: {indicators['hypervolume']:.3f}, Sharpe Ratio: {indicators['sharpe_ratio']:.3f}")
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "PESA")"""

    # Run E-MOEA algorithm    
    e = 0.00458 * 1.75
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, crossover_rate - 0.1, mutation_rate, generations, e)
    final_population = e_moea.evolve()
    indicators = calculate_performance(final_population, returns, cov_matrix)
    print(f"Hypervolume: {indicators['hypervolume']:.3f}, Sharpe Ratio: {indicators['sharpe_ratio']:.3f}")
    print_pareto_front(final_population, returns, cov_matrix, cardinality, "e-MOEA")
            