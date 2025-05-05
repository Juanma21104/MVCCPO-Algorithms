import random
import time
from matplotlib import pyplot as plt
import numpy as np
import algorithms.utils as utils

class NSGA2Pro:

    def __init__(self, N_arc, N_pop, num_assets, returns, cov_matrix, cardinality, mutation_rate, generations):
        self.N_arc = N_arc # Archive population size (A_0)
        self.N_pop = N_pop # Usual population size (B_0)
        self.cardinality = cardinality # Number of assets in the portfolio
        self.num_assets = num_assets
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        self.population_A = utils.initialize_population(self.N_arc, self.num_assets, self.cardinality)[0] # Archive population (A_0)
        self.population_B = utils.initialize_population(self.N_pop, self.num_assets, self.cardinality)[1] # Usual population (B_0)

    def fast_non_dominated_sort(self, population):
        """
        Perform fast non-dominated sorting on the population.

        Parameters:
        - population: A 2D array representing the population of portfolios.

        Returns:
        - fronts: A list of fronts, where each front contains the indices of the individuals in that front.
        
        """
        fronts = [[]] # List of fronts
        population_size = len(population) # Size of the population 
        dominance_count = np.zeros(population_size) # A 2D array that count of how many individuals dominate individual i
        dominated_sets = [[] for _ in range(population_size)] # List of dominated sets for each individual

        matrix_ret_risks = utils.precompute_objectives(population, self.returns, self.cov_matrix)
        
        for i in range(population_size): # Index and portfolio
            for j in range(population_size): # Compare with all other portfolios
                if utils.dominates(matrix_ret_risks, i, j): # If i dominates j
                    dominated_sets[i].append(j)
                elif utils.dominates(matrix_ret_risks, j, i): # If j dominates i
                    dominance_count[i] += 1
            if dominance_count[i] == 0: # If no one dominates this individual, it belongs to the first front
                fronts[0].append(i)

        i = 0
        while fronts[i]:
            next_front = [] # Next front to be filled
            for p in fronts[i]: # For each individual in the current front
                for q in dominated_sets[p]: # For each individual that this individual dominates
                    dominance_count[q] -= 1
                    if dominance_count[q] == 0: # If this individual is no longer dominated
                        next_front.append(q)
            i += 1
            fronts.append(next_front) # Add the next front to the list of fronts

        
        return fronts[:-1] # Remove the last empty front
    
    def crowding_distance_assignment(self, front, population):
        """
        Assign crowding distance to each individual in the front.

        Parameters:
        - front: A list of indices representing the individuals in the front.

        Returns:
        - distances: A 1D array representing the crowding distance of each individual in the front.
        """

        distances = np.zeros(len(front)) # Initialize distances to zero
        for m in range(2): # For each objective (expected return and risk)
            front.sort(key=lambda x: utils.evaluate(population[x], self.returns, self.cov_matrix)[m]) # Sort the front by the m-th objective
            distances[0] = distances[-1] = np.inf # Assign infinite distance to the limits
            min_val = utils.evaluate(population[front[0]], self.returns, self.cov_matrix)[m]
            max_val = utils.evaluate(population[front[-1]], self.returns, self.cov_matrix)[m]
            for i in range(1, len(front) - 1): # For each individual in the front (except the limits)
                if max_val - min_val == 0: # Avoid division by zero
                    distances[i] = 0
                else:
                    # Calculate the crowding distance, which is the normalized distance between the two neighbors
                    distances[i] += (utils.evaluate(population[front[i + 1]], self.returns, self.cov_matrix)[m] -
                                 utils.evaluate(population[front[i - 1]], self.returns, self.cov_matrix)[m]) / (max_val - min_val)
                 
        return distances

    def selection(self, fronts, population, population_size):
        """
        Select individuals for the next generation based on non-dominated sorting and crowding distance.
        
        Parameters:
        - fronts: A list of fronts, where each front contains the indices of the individuals in that front.

        Returns:
        - new_population: A 2D array representing the selected individuals for the next generation.

        """
        new_population = [] # Initialize the new population (index of individuals)
        for front in fronts:
            if len(new_population) + len(front) <= population_size: # If the new population size does not exceed the limit
                new_population.extend(front)
            else:
                distances = self.crowding_distance_assignment(front, population) # Calculate the crowding distance for the front
                sorted_front = sorted(zip(front, distances), key=lambda x: -x[1]) # Zip the front with the distances and sort by distance (highest to lowest (-x[1]))
                new_individuals = sorted_front[:population_size - len(new_population)] # Select the best individuals based on distance
                new_population.extend([x[0] for x in new_individuals]) # Add the selected individuals to the new population
        return np.array([population[i] for i in new_population]) # Convert indices to actual individuals

    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """
        new_population = []
        fronts = self.fast_non_dominated_sort(population)

        i = 0
        distances = []
        for front in fronts:
            distances.append(self.crowding_distance_assignment(front, population))
            i += 1
        for _ in range(self.N_pop):
            parents = self.tournament(population, fronts, distances)
            child = utils.crossover(population[parents[0]], population[parents[1]], self.num_assets, self.cardinality)
            child = utils.mutation(child, self.mutation_rate)
            new_population.append(child)
        return np.array(new_population)
    
    def tournament(self, population, fronts, crowding_distance):
        """
        Perform tournament selection.
        
        Parameters:
        - fronts: A list of fronts, where each front contains the indices of the individuals in that front.
        - crowding_distance: A list of crowding distances for each individual.
        
        Returns:
        - selected_population: The selected individuals.
        """
        selected_population = []
        for _ in range(2):
            candidates = random.sample(range(len(population)), 2)
            front_candidate1 = self.getIndexFront(fronts, candidates[0])
            front_candidate2 = self.getIndexFront(fronts, candidates[1])
            
            pos_in_front_1 = fronts[front_candidate1].index(candidates[0])
            pos_in_front_2 = fronts[front_candidate2].index(candidates[1])
            if front_candidate1 == front_candidate2:
                if crowding_distance[front_candidate1][pos_in_front_1] > crowding_distance[front_candidate2][pos_in_front_2]:
                    selected_population.append(candidates[0])
                else:
                    selected_population.append(candidates[1])
            else:
                if front_candidate1 < front_candidate2:
                    selected_population.append(candidates[0])
                else:
                    selected_population.append(candidates[1])
        return np.array(selected_population)

    def getIndexFront(self, front, index):
        for i, subarray in enumerate(front):
            if index in subarray:
                return i

        return -1


    def best(self, population_A, population_B):
        """
        Select the best individuals from the archive population and the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - best_population: The combined best individuals from both populations.

        """
        combined_population = np.vstack((population_A, population_B)) # Combine the two populations
        fronts = self.fast_non_dominated_sort(combined_population)
        selected_population = self.selection(fronts, combined_population, self.N_arc)
        return selected_population

    def evolve(self):
        """
        Evolve the population over a number of generations using NSGA-II.

        Returns:
        - population: The final population after evolution.

        """
        i = 0
        started_time = time.time()
        
        while i < self.generations:
            print("Generation: ", i)
            self.population_A = self.best(self.population_A, self.population_B)
            self.population_B = self.vary(self.population_A)

            """self.population_A = self.best(self.population_A, self.population_B)
            offsprings = self.vary(self.population_B)
            self.population_B = self.best(self.population_B, offsprings)"""
            i += 1            

        end_time = time.time()
        elapsed_time = end_time - started_time
        self.population_A = self.best(self.population_A, self.population_B)
        print(f"Execution time: {elapsed_time:.2f} seconds")

        return self.population_A
    

    def plot_pareto_front(self):
        pareto_front = self.fast_non_dominated_sort(self.population_A)[0]
        pareto_points = np.array([utils.evaluate(self.population_A[i], self.returns, self.cov_matrix) for i in pareto_front])

        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_points[:, 1], pareto_points[:, 0], color='red', label='Pareto Front')
        plt.xlabel('Variance')
        plt.ylabel('Mean')
        plt.title('Pareto front - Portfolio Optimization')
        plt.legend()
        plt.grid()
        plt.show()