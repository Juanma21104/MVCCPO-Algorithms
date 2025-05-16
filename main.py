from algorithms.nsga2 import NSGA2
from algorithms.spea2 import SPEA2
from algorithms.npga2 import NPGA2
from algorithms.pesa import PESA
from algorithms.e_moea import E_MOEA
from algorithms.soea import SOEA
from algorithms.data_loader import load_dataset

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

    trade_off_coeff = 0.5
 
    # Run NSGA-II 
    """nsga2 = NSGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    nsga2.evolve()
    nsga2.plot_pareto_front()"""


    # Run SPEA-II algorithm
    """spea2 = SPEA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations)
    spea2.evolve()
    spea2.plot_pareto_front()"""


    # Run NPGA2 algorithm    
    """npga2 = NPGA2(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, tournament_size, niche_radius)
    npga2.evolve()
    npga2.plot_pareto_front()"""


    # Run PESA algorithm    
    """pesa = PESA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate - 0.1, generations)
    pesa.evolve()
    pesa.plot_pareto_front()"""


    # Run E-MOEA algorithm    
    e = 0.00458 * 1.1
    e_moea = E_MOEA(N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate - 0.1, generations, e)
    e_moea.evolve()
    e_moea.plot_pareto_front()
    

    # Run SOEA algorithm    
    """soea = SOEA(100, 1, num_assets, returns, cov_matrix, cardinality, mutation_rate, 100000, trade_off_coeff)
    soea.evolve()
    print("Best individual:", soea.get_best_individual())
    ret, risk = utils.evaluate(soea.get_best_individual(), returns, cov_matrix)
    print("Returns and risks:", ret, risk)"""



    








    
            