import algorithms.utils as utils
import time
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class E_MOEA:
    def __init__(self, N_pop, N_arc, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, e):
        self.N_pop = N_pop
        self.N_arc = N_arc
        self.num_assets = num_assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.cardinality = cardinality
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.e = e

        populations = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)

        self.population_A = populations[0]
        self.population_B = populations[1]


    def is_not_dominated(self, idx_ind_B, matrix_ret_risks_B, matrix_ret_risks_A):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - individual: The individual to check.
        - population: The population to check against.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """
        ret_B, risk_B = matrix_ret_risks_B.T[idx_ind_B]

        for idx_ind_A in range(len(matrix_ret_risks_A.T)):
            ret_A, risk_A = matrix_ret_risks_A.T[idx_ind_A]
            if self.dominates((ret_A, risk_A), (ret_B, risk_B)):
                return False
        return True

    def dominates_any(self, idx_ind_B, matrix_ret_risks_B, matrix_ret_risks_A):
        """
        Check if an individual dominates any individual in the population.
        
        Parameters:
        - individual: The individual to check.
        - population: The population to check against.
        
        Returns:
        - True if the individual dominates any individual in the population, False otherwise.
        """
        if(len(matrix_ret_risks_A.T) == 0):
            return True

        ret_B, risk_B = matrix_ret_risks_B.T[idx_ind_B]
        for ind2 in matrix_ret_risks_A.T:
            ret_A, risk_A = ind2
            
            if self.dominates((ret_A, risk_A), (ret_B, risk_B)):
                return True

        return False

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
    
    def smallest_distance_bigger_than_e(self, ind, points):
        """
        Check if the smallest distance between an individual and any individual in the population is bigger than e.
        
        Parameters:
        - individual: The individual to check.
        - population: The population to check against.
        
        Returns:
        - True if the smallest distance is bigger than e, False otherwise.
        """
        if len(points.T) == 0:
            return True, np.array([])
        ind_obj = np.array([ind])
        points = (self.normalize_objectives(points)).T
        #distances = cdist(ind_obj, points, 'euclidean')[0]
        distances = np.linalg.norm(points - ind_obj, axis=1)

        return np.min(distances) > self.e, distances

    def improve_optimum(self, evaluated_individual, points):
        if(len(points) == 0):
            return False
        if(evaluated_individual[0] > np.max(points[:, 0]) or evaluated_individual[1] < np.min(points[:, 1])):
            return True
        return False

    def remove_dominated_solutions(self, population, matrix_ret_risks):
        dominance_count = np.zeros(len(population))
        non_dominated = []
        for i in range(len(population)):
            for j in range(len(population)):
                if utils.dominates(matrix_ret_risks, j, i):
                    dominance_count[i] += 1
            if dominance_count[i] == 0:
                non_dominated.append(population[i])
        return np.array(non_dominated)

    def dominates(self, ret_risk1, ret_risk2):
        ret1, risk1 = ret_risk1
        ret2, risk2 = ret_risk2
        return (ret1 >= ret2 and risk1 <= risk2) and (ret1 > ret2 or risk1 < risk2)
    
    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

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


    def update(self, population_A, population_B):
        matrix_ret_risks_B = utils.precompute_objectives(population_B, self.returns, self.cov_matrix)

        for idx_ind_B in range(len(population_B)):
            added = False
            matrix_ret_risks_A = utils.precompute_objectives(population_A, self.returns, self.cov_matrix)
            condition2, distances = self.smallest_distance_bigger_than_e(matrix_ret_risks_B.T[idx_ind_B], matrix_ret_risks_A)

            if self.dominates_any(idx_ind_B, matrix_ret_risks_B, matrix_ret_risks_A):
                population_A = np.vstack((population_A, population_B[idx_ind_B]))
                added = True
            elif self.is_not_dominated(idx_ind_B, matrix_ret_risks_B, matrix_ret_risks_A):
                if condition2:
                    population_A = np.vstack((population_A, population_B[idx_ind_B]))
                    added = True
            
            if self.improve_optimum(matrix_ret_risks_B.T[idx_ind_B], matrix_ret_risks_A.T):
                if not added:
                    population_A = np.vstack((population_A, population_B[idx_ind_B]))
                idxs_to_remove = []
                for idx, dist in enumerate(distances):
                    if dist < self.e:
                        idxs_to_remove.append(idx)
                population_A = np.delete(population_A, idxs_to_remove, axis=0)
                    
        print("Population A size: ", len(population_A))
        matrix_ret_risks_A = utils.precompute_objectives(population_A, self.returns, self.cov_matrix)
        population_A = self.remove_dominated_solutions(population_A, matrix_ret_risks_A)
        print("Population A size after removing dominated: ", len(population_A))
    
        return population_A


    def evolve(self):
        """
        Evolve the population over a number of generations using PESA.

        Returns:
        - population_A: The final population after evolution.
        """

        i = 0
        started_time = time.time()
        while i < self.generations:
            print(f"Generation: {i}")
            self.population_A = self.update(self.population_A, self.population_B)
            self.population_B = self.vary(self.population_A)
            i += 1

        end_time = time.time()
        elapsed_time = end_time - started_time
        print(f"Execution time: {elapsed_time:.2f} seconds")
        self.population_A = self.update(self.population_A, self.population_A)

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