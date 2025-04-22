import algorithms.utils as utils
import time
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

class SPEA2:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets # Number of assets
        self.returns = returns # Returns of the assets
        self.cov_matrix = cov_matrix # Covariance matrix of the assets
        self.mutation_rate = mutation_rate # Mutation rate
        self.generations = generations # Number of genetations
        
        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[0] # Archive population (A_0)
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[1] # Usual population (B_0)

    def precompute_objectives(self, population):
        """
        Precompute the returns and risks for all individuals in the population.
        
        Parameters:
        - population: A 2D array representing the population of portfolios.
        
        Returns:
        - matrix: A 2D array containing the returns and risks of the population.
        """
        N = len(population)
        matrix = np.zeros((2, N))
        for i in range(N):
            ret, risk = utils.evaluate(population[i], self.returns, self.cov_matrix)
            matrix[0][i] = ret
            matrix[1][i] = risk
        return matrix
    
    def dominates(self, matrix_ret_risks, ind1, ind2):
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

    def raw_fitness(self, population, matrix_ret_risks):
        N = len(population)
        dominance_count = np.zeros(N) # A 2D array that count of how many individuals dominate individual i
        dominated_sets = [[] for _ in range(N)] # List of dominated sets for each individual
        
        for i in range(N): # Index and portfolio
            for j in range(N): # Compare with all other portfolios
                if self.dominates(matrix_ret_risks, i, j): # If i dominates j
                    dominated_sets[i].append(j)
                elif self.dominates(matrix_ret_risks, j, i): # If j dominates i
                    dominance_count[i] += 1

        raw_fitness_values = np.zeros(N)
        for i in range(N):
            raw_fitness_values[i] = dominance_count[i] / (len(dominated_sets[i]) if len(dominated_sets[i]) > 0 else 1)

        return raw_fitness_values
        

    def calculate_density(self, matrix_ret_risks, k=1):
        """
        Calculate the density of each individual in the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        - k: Number of neighbors to consider.
        
        Returns:
        - D: A 1D array representing the density of each individual in the population.
        """
        points = matrix_ret_risks.T  # Shape (N, 2) 
        distances = cdist(points, points) # Compute distance between all points
        np.fill_diagonal(distances, np.inf) # Set diagonal to infinity to avoid self-comparison
        kth_distances = np.partition(distances, k, axis=1)[:, k] # k-th smallest distance for each point
        return 1.0 / (kth_distances + 2.0) # Density is the inverse of the k-th smallest distance

    def calculate_total_fitness(self, population, k=1, return_matrix=False):
        """Calculate the total fitness of each individual in the population.
        
        Parameters:
        - population: A 2D array representing the population of portfolios.
        - k: Number of neighbors to consider.
        
        Returns:
        - F: A 1D array representing the total fitness of each individual in the population.
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        """
        matrix_ret_risks = self.precompute_objectives(population)
        R = self.raw_fitness(population, matrix_ret_risks)
        D = self.calculate_density(matrix_ret_risks, k)
        F = R + D
        return (F, matrix_ret_risks) if return_matrix else F

    def compute_dominance_matrix(self, matrix_ret_risks):
        """
        Compute the dominance matrix for the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - domination_matrix: A 2D array representing the dominance matrix.
        """
        N = matrix_ret_risks.shape[1]
        domination_matrix = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                if self.dominates(matrix_ret_risks, i, j):
                    domination_matrix[i, j] = True
                elif self.dominates(matrix_ret_risks, j, i):
                    domination_matrix[j, i] = True
        return domination_matrix

    def update(self, population_A, population_B):
        """
        Select the best individuals from the archive population and the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - new_archive: The combined best population from both populations.

        """
        combined = np.vstack((population_A, population_B))
        fitness, matrix_ret_risks = self.calculate_total_fitness(combined, return_matrix=True)


        dom_matrix = self.compute_dominance_matrix(matrix_ret_risks)
        non_dominated_indices = [i for i in range(len(combined)) if not np.any(dom_matrix[:, i])]
        new_archive = [combined[i] for i in non_dominated_indices]

        # If there are more than allowed, truncate
        if len(new_archive) > self.N_arc:
            new_archive = self.truncate(new_archive, self.N_arc)
        
        # If there are less, complete with the best dominated
        elif len(new_archive) < self.N_arc:
            dominated_indices = [i for i in range(len(combined)) if i not in non_dominated_indices]
            dominated_sorted = sorted(dominated_indices, key=lambda i: fitness[i])
            for i in dominated_sorted:
                if len(new_archive) >= self.N_arc:
                    break
                new_archive.append(combined[i])

        return new_archive

    def truncate(self, population, N_arc):
        """
        Truncate the population by removing the least diverse solutions.

        Parameters:
        - population: The population to truncate.
        - N_arc: The target size of the population.

        Returns:
        - truncated_population: The truncated population.
        """
        N = len(population)
        matrix_ret_risks = self.precompute_objectives(population)
        points = matrix_ret_risks.T  # shape (N, 2)

        # Compute pairwise distances
        distance_matrix = cdist(points, points)
        np.fill_diagonal(distance_matrix, np.inf)

        # Indices of individuals still in the archive
        remaining = list(range(N))
        sort_distance_matrix = np.sort(distance_matrix, axis=1)
        nearest_distances = sort_distance_matrix[:, 0]

        while len(remaining) > N_arc:
            min_indexes = np.where(nearest_distances == np.min(nearest_distances))[0]
            to_remove = None
            if(len(min_indexes) > 1):
                k = 1
                
                while True:
                    if(k >= N):
                        to_remove = min_indexes[0]
                        break

                    index_value = np.column_stack((min_indexes, sort_distance_matrix[min_indexes, k]))
                    min_value = np.min(index_value[:, 1])
                    min_index2 = np.where(index_value[:, 1] == min_value)[0]

                    if(len(min_index2) == 1):
                        to_remove = index_value[min_index2[0]][0]
                        break
                    k += 1
            else:
                to_remove = min_indexes[0]

            nearest_distances[int(to_remove)] = np.inf
            remaining.remove(int(to_remove))
        
        return [population[i] for i in remaining]


    def binary_tournament(self, population, fitness):
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


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fitness = self.calculate_total_fitness(population)
        for _ in range(self.N_pop):
            parent1 = self.binary_tournament(population, fitness)
            parent2 = self.binary_tournament(population, fitness)
            while utils.evaluate(parent1, self.returns, self.cov_matrix) == utils.evaluate(parent2, self.returns, self.cov_matrix):
                parent2 = self.binary_tournament(population, fitness)
            child = utils.crossover(parent1, parent2, self.num_assets, self.cardinality)
            child = utils.mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)


    def evolve(self):
        """
        Evolve the population over a number of generations using NSGA-II.

        Returns:
        - population: The final population after evolution.

        """

        i = 0
        started_time = time.time()
        while i < self.generations:
            print(f"Generation: {i}")
            self.population_A = self.update(self.population_A, self.population_B)
            offsprings = self.vary(self.population_B)
            self.population_B = self.update(self.population_B, offsprings)
            i += 1

        end_time = time.time()
        elapsed_time = end_time - started_time
        self.population_A = self.update(self.population_A, self.population_B)
        print(f"Execution time: {elapsed_time:.2f} seconds")

        return self.population_A

    
    def plot_pareto_front(self):
        pareto_points = self.precompute_objectives(self.population_A)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_points[1, :], pareto_points[0, :], color='red', label='Pareto Front')
        plt.xlabel('Variance')
        plt.ylabel('Mean')
        plt.title('Pareto front - Portfolio Optimization')
        plt.legend()
        plt.grid()
        plt.show()