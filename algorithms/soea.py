import algorithms.utils as utils
import time
import numpy as np
import random

class SOEA:
    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, trade_off_coeff):
        self.N_arc = N_arc
        self.N_pop = N_pop
        self.cardinality = cardinality
        self.num_assets = num_assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.trade_off_coeff = trade_off_coeff

        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[1]
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[0]

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

    def fitness(self, matrix_ret_risks):
        """
        Calculate the fitness of the population.
        
        Parameters:
        - matrix_ret_risks: A 2D array containing the returns and risks of the population.
        
        Returns:
        - fitness: A 1D array containing the fitness of the population.
        """
        fitness = []
        for i in range(matrix_ret_risks.shape[1]):
            fitness.append(self.trade_off_coeff * matrix_ret_risks[1][i] - (1 - self.trade_off_coeff) * matrix_ret_risks[0][i])
        return fitness
        
    def update(self, population_A, individual, fitness):
        """
        Select the best individuals from the archive population and the current population.

        Parameters:
        - population_A: The archive population (numpy array).
        - individual: The individual to add to the archive population.
        - fitness: A 1D array representing the fitness of each individual in the population.

        Returns:
        - new_archive: The updated archive population (numpy array).
        """
        idx_to_remove = np.argmax(fitness)
        population_A[idx_to_remove] = individual

        return population_A

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

    def vary(self, population, fitness):
        """
        Perform crossover and mutation on the population.
        
        Parameters:
        - population: A 2D array representing the population of portfolios.
        - fitness: A 1D array representing the fitness of each individual in the population.
        
        Returns:
        - offsprings: A 2D array containing the offsprings.
        """
        parent1 = self.binary_tournament(population, fitness)
        parent2 = self.binary_tournament(population, fitness)
        while utils.evaluate(parent1, self.returns, self.cov_matrix) == utils.evaluate(parent2, self.returns, self.cov_matrix):
            parent2 = self.binary_tournament(population, fitness)
        child = utils.crossover(parent1, parent2, self.num_assets, self.cardinality)
        child = utils.mutation(child, self.mutation_rate)
        return child

    def evolve(self):
        t = 0
        start = time.time()
        while t < self.generations:
            print(f"Generation: {t}")
            matrix_ret_risks = self.precompute_objectives(self.population_A)
            fitness = self.fitness(matrix_ret_risks)
            self.population_B = self.vary(self.population_A, fitness)
            self.population_A = self.update(self.population_A, self.population_B, fitness)
            t += 1
        print(f"Execution time: {time.time() - start:.2f}s")
        return self.population_A

    def get_best_individual(self):
        """
        Returns the best individual in the population.
        """
        matrix_ret_risks = self.precompute_objectives(self.population_A)
        fitness = self.fitness(matrix_ret_risks)
        idx = np.argmin(fitness)
        return self.population_A[idx]

        
        