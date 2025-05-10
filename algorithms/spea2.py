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
                if utils.dominates(matrix_ret_risks, i, j):
                    domination_matrix[i, j] = True
                elif utils.dominates(matrix_ret_risks, j, i):
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
        fitness, matrix_ret_risks = utils.calculate_total_fitness(combined, self.returns, self.cov_matrix, return_matrix=True)


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
        matrix_ret_risks = utils.precompute_objectives(population, self.returns, self.cov_matrix)
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

    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.
        - fitness: The fitness values of the population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fitness = utils.calculate_total_fitness(population, self.returns, self.cov_matrix)
        for _ in range(self.N_pop):
            parent1 = utils.binary_tournament(population, fitness)
            parent2 = utils.binary_tournament(population, fitness)
            while utils.evaluate(parent1, self.returns, self.cov_matrix) == utils.evaluate(parent2, self.returns, self.cov_matrix):
                parent2 = utils.binary_tournament(population, fitness)
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
            #self.population_A = self.update(self.population_A, self.population_B)
            #self.population_B = self.vary(self.population_A)
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
        pareto_points = utils.precompute_objectives(self.population_A, self.returns, self.cov_matrix)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_points[1, :], pareto_points[0, :], color='red', label='Pareto Front')
        plt.xlabel('Variance')
        plt.ylabel('Mean')
        plt.title('Pareto front - Portfolio Optimization')
        plt.legend()
        plt.grid()
        plt.show()