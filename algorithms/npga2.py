import algorithms.utils as utils

class NPGA2:

    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, niche_radius):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        self.niche_radius = niche_radius
        
        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[0] # Archive population (A_0)
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[1] # Usual population (B_0)

    
    def niche_count(self, matrix_distance, ind_i, ind_j):
        if matrix_distance[ind_i][ind_j] >= self.niche_radius:
            return 0
        else:
            return (1 - ( matrix_distance[i]/ self.niche_radius)).sum()


    