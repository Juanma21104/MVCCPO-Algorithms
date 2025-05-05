import algorithms.utils as utils
import time
import numpy as np
import random
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

class NPGA2:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, tdom, rsh):
        self.N_arc = N_arc
        self.N_pop = N_pop
        self.cardinality = cardinality
        self.num_assets = num_assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.tdom = tdom  # Tournament size
        self.rsh = rsh    # Niche radius

        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[0]
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[1]

    def dominance_rank(self, matrix_ret_risks):
        """
        Calculate the dominance rank of each individual in the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - rank: A 1D array containing the dominance rank of each individual.
        """
        N = matrix_ret_risks.shape[1]
        rank = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i != j:
                    if utils.dominates(matrix_ret_risks, j, i):
                        rank[i] += 1
        return rank

    def normalize_objectives(self, matrix_ret_risks):
        """
        Normalize the objectives in the matrix.
        
        Parameters:
        - matrix: A 2D array containing the returns and risks of the population.
        
        Returns:
        - norm: A 2D array containing the normalized returns and risks.
        """
        norm = np.zeros_like(matrix_ret_risks)
        for i in range(2):
            fmin, fmax = matrix_ret_risks[i].min(), matrix_ret_risks[i].max()
            #print("Fmin: ", fmin, "   Fmax: ", fmax)
            norm[i] = (matrix_ret_risks[i] - fmin) / (fmax - fmin + 1e-10)
        return norm

    def metropolitan_distance(self, matrix_ret_risks):
        norm_matrix = self.normalize_objectives(matrix_ret_risks).T
        """N = norm_matrix.shape[0]
        distances = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.sum(np.abs(norm_matrix[i] - norm_matrix[j]))"""
        distances = cdist(norm_matrix, norm_matrix, metric='cityblock')
        return distances

    def niche_count(self, distances, next_gen, i):
        """
        Calculate the niche count of an individual.
        
        Parameters:
        - distances: A 2D array containing the metropolitan distances between individuals.
        - next_gen: The next generation of individuals.
        - i: Index of the individual.
        
        Returns:
        - count: The niche count of the individual.
        """
        count = 0
        for j in range(len(next_gen)):
            if distances[i, j] < self.rsh:
                count += 1 - distances[i, j] / self.rsh
        return count

    def update(self, population_A, population_B):
        combined = np.vstack((population_A, population_B))
        matrix_ret_risks = utils.precompute_objectives(combined, self.returns, self.cov_matrix)
        rank = self.dominance_rank(matrix_ret_risks)
        distances = self.metropolitan_distance(matrix_ret_risks)

        archive = []
        available = set(range(len(combined)))
        while len(archive) < self.N_arc:
            candidates = random.sample(list(available), self.tdom)
            #candidates = random.sample(range(len(combined)), self.tdom)
            best_rank = min(rank[i] for i in candidates)
            best = [i for i in candidates if rank[i] == best_rank]
            if len(best) > 1:
                niche_counts = [self.niche_count(distances, archive, i) for i in best]
                best_idx = best[np.argmin(niche_counts)]
            else:
                best_idx = best[0]

            #if not any(np.array_equal(combined[best_idx], a) for a in archive):
            #    archive.append(combined[best_idx])
            archive.append(combined[best_idx])
            available.remove(best_idx)
        return archive

    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.
        
        Parameters:
        - population: A 2D array representing the population of portfolios.
        
        Returns:
        - new_population: The new population after genetic operations.
        """
        new_population = []
        ranks = self.dominance_rank(utils.precompute_objectives(population, self.returns, self.cov_matrix))
        for _ in range(self.N_pop):
            p1 = utils.binary_tournament(population, ranks)
            p2 = utils.binary_tournament(population, ranks)
            while utils.evaluate(p1, self.returns, self.cov_matrix) == utils.evaluate(p2, self.returns, self.cov_matrix):
                p2 = utils.binary_tournament(population, ranks)
            child = utils.crossover(p1, p2, self.num_assets, self.cardinality)
            child = utils.mutation(child, self.mutation_rate)
            new_population.append(child)
        return new_population

    def evolve(self):
        t = 0
        start = time.time()
        while t < self.generations:
            print(f"Generation: {t}")
            if t != 0:
                self.population_A = self.update(self.population_A, self.population_B)
            else:
                self.population_A = self.population_B
            
            offsprings = self.vary(self.population_A)
            self.population_B = self.update(self.population_B, offsprings)

            #self.population_B = self.vary(self.population_A)
            t += 1
        self.population_A = self.update(self.population_A, self.population_B)
        print(f"Execution time: {time.time() - start:.2f}s")
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
