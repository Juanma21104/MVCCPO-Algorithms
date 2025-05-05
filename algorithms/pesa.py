import time
import numpy as np
import algorithms.utils as utils
import matplotlib.pyplot as plt
from algorithms import nsga2_pro

class PESA:
    def __init__(self, N_pop, N_arc, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations, grid_divisions=10):
        self.N_pop = N_pop
        self.N_arc = N_arc
        self.num_assets = num_assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.cardinality = cardinality
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.grid_divisions = grid_divisions

        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[0]
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[1]


    def evaluate_population(self, population):
        """
        Evaluate the population.
        
        Parameters:
        - population: The population to evaluate.
        
        Returns:
        - np.array: An array of risk and returns for each individual in the population.
        """
        return np.array([utils.evaluate(ind, self.returns, self.cov_matrix) for ind in population])

    """def assign_hypergrid(self, points):
        # A point is (return, risk), so (y,x)
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)

        cell_size = (max_vals - min_vals) / self.grid_divisions

        hypergrid = [ [] for _ in range(self.grid_divisions * self.grid_divisions)]

        iter = 0
        min_width = min_vals[1]
        min_width_reset = min_width
        max_width = cell_size[1] + min_width
        max_width_reset = max_width

        min_heigh = min_vals[0]
        max_heigh = cell_size[0] + min_heigh


        print("Min width: ", min_width)
        print("Min heigh: ", min_heigh)

        print("Total points: ", len(points))

        total = 0

        for i in range(self.grid_divisions):
            for j in range(self.grid_divisions):
                for index, point in enumerate(points):
                    if(j == 0 and i == 0):
                        if min_width <= point[1] <= max_width and min_heigh <= point[0] <= max_heigh:
                            total += 1
                            hypergrid[iter].append(index)
                    elif min_width < point[1] <= max_width and min_heigh < point[0] <= max_heigh:
                        total += 1
                        hypergrid[iter].append(index)
                
                min_width = max_width
                max_width += cell_size[1]
                iter += 1
            min_heigh = max_heigh
            max_heigh += cell_size[0]
            min_width = min_width_reset
            max_width = max_width_reset
                
        flattened = [num for sublist in hypergrid for num in sublist]

        # Crear el conjunto del rango completo
        full_range = set(range(250))

        # Crear el conjunto de los números presentes
        present_numbers = set(flattened)

        # Calcular los que faltan
        missing_numbers = full_range - present_numbers

        # Convertir a lista si quieres
        missing_numbers = sorted(list(missing_numbers))

        print("Missing numbers: ", missing_numbers)

        for i in missing_numbers:
            print("Point: ", points[i])

        print("Max width: ", max_width)
        print("Max heigh: ", max_heigh)

        print("Total: ", total)
        return hypergrid"""

    def assign_hypergrid(self, points):
        """
        Assign points to a hypergrid.
        
        Parameters:
        - points: A numpy array of shape (N, 2), where each point is (return, risk)
        
        Returns:
        - hypergrid: A list of lists with point indices assigned to grid cells
        """

        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        cell_size = (max_vals - min_vals) / self.grid_divisions

        hypergrid = [[] for _ in range(self.grid_divisions * self.grid_divisions)]

        for idx, point in enumerate(points):
            row = int((point[0] - min_vals[0]) / cell_size[0])
            col = int((point[1] - min_vals[1]) / cell_size[1])

            # Clamp to avoid going out of bounds due to precision
            row = min(row, self.grid_divisions - 1)
            col = min(col, self.grid_divisions - 1)

            cell_index = row * self.grid_divisions + col
            hypergrid[cell_index].append(idx)

        return hypergrid

    def get_individual_fitness(self, hypergrid, ind):
        """
        Get the fitness of an individual.
        
        Parameters:
        - hypergrid: The hypergrid.
        - ind: The individual.
        
        Returns:
        - length: The fitness of the individual.
        """
        for sublist in hypergrid:
            if ind in sublist:
                length = len(sublist)
                break
        return length


    def fitness(self, population):
        points = self.evaluate_population(population)
        hypergrid = self.assign_hypergrid(points)
        fitness = np.zeros(len(population))

        for i in range(len(population)):
            fitness[i] = self.get_individual_fitness(hypergrid, i)

        return fitness


    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """

        new_population = []
        fitness = self.fitness(population)
        for _ in range(self.N_pop):
            parent1 = utils.binary_tournament(population, fitness)
            parent2 = utils.binary_tournament(population, fitness)
            while utils.evaluate(parent1, self.returns, self.cov_matrix) == utils.evaluate(parent2, self.returns, self.cov_matrix):
                parent2 = utils.binary_tournament(population, fitness)
            child = utils.crossover(parent1, parent2, self.num_assets, self.cardinality)
            child = utils.mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)


    def is_not_dominated(self, individual, population):
        """
        Check if an individual is not dominated by any individual in the population.
        
        Parameters:
        - individual: The individual to check.
        - population: The population to check against.
        
        Returns:
        - True if the individual is not dominated, False otherwise.
        """
        for ind2 in population:
            if self.dominates(ind2, individual):
                return False
        return True

    def dominates(self, individual1, individual2):
        ret1, risk1 = utils.evaluate(individual1, self.returns, self.cov_matrix)
        ret2, risk2 = utils.evaluate(individual2, self.returns, self.cov_matrix)
        return (ret1 >= ret2 and risk1 <= risk2) and (ret1 > ret2 or risk1 < risk2)


    def update(self, population_A, population_B):
        """
        Update the archive population with the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - population_A: The updated archive population.
        """
        
        for ind in population_B:
            if (self.is_not_dominated(ind, population_A)):
                population_A = np.vstack((population_A, ind))
                if len(population_A) > self.N_arc:
                    individual_to_remove = np.argmax(self.fitness(population_A))
                    population_A = np.delete(population_A, individual_to_remove, axis=0)
            else:
                if (self.is_not_dominated(ind, population_B)):
                    for idx, ind2 in enumerate(population_A):
                        if self.dominates(ind, ind2):
                            population_A = np.delete(population_A, idx, axis=0)
                            population_A = np.vstack((population_A, ind))
                            break

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
        self.population_A = self.update(self.population_A, self.population_B)

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