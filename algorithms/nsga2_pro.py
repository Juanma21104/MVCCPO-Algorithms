import random
import time
from matplotlib import pyplot as plt
import numpy as np

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
        
        self.population_A = self.initialize_population()[0] # Archive population (A_0)
        self.population_B = self.initialize_population()[1] # Usual population (B_0)


    def initialize_population(self):
        """
        Initialize the population with random portfolios.


        Returns:
        - population A: A 2D empty array
        - population B: A 2D array representing the initial population of portfolios.
        """
        population_A = np.empty(self.num_assets)
        population_B = []
        for _ in range(self.N_pop):
            individual = np.zeros(self.num_assets)
            selected_assets = random.sample(range(self.num_assets), self.cardinality) # Select random assets to include in the portfolio
            for asset in selected_assets:
                value = random.uniform(0, 1)
                while value == 0:
                    value = random.uniform(0, 1)
                individual[asset] = value
            individual /= individual.sum()
            population_B.append(individual)
        return population_A, np.array(population_B)
    
    def evaluate(self, individual):
        """
        Evaluate the portfolio represented by the individual.
        
        Parameters:
        - individual: A portfolio represented as a vector of weights.
        
        Returns:
        - expected_return: Expected return of the portfolio.
        - risk: Risk (variance) of the portfolio.
        """
        expected_return = np.dot(individual, self.returns) # Expected return is the dot product of weights and expected returns
        risk = np.dot(individual.T, np.dot(self.cov_matrix, individual)) # Risk = w T * cov_matrix * w
        return expected_return, risk

    def update(self, population_A, population_B):
        """
        Update the archive population with the current population.

        Parameters:
        - population_A: The archive population.
        - population_B: The current population.

        Returns:
        - updated_population_A: The updated archive population.
        
        """
        combined_population = np.vstack((population_A, population_B))
        fronts = self.fast_non_dominated_sort(combined_population)
        selected_population = self.selection(fronts, combined_population, self.N_arc)
        return selected_population

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

        matrix_ret_risks = np.zeros((2,population_size)) # Matrix to store the returns and risks of the population
        for i in range(population_size):
            returns, risk = self.evaluate(population[i])
            matrix_ret_risks[1][i] = risk # Store the risk in the matrix
            matrix_ret_risks[0][i] = returns
        
        for i in range(population_size): # Index and portfolio
            for j in range(population_size): # Compare with all other portfolios
                if self.dominates(matrix_ret_risks, i, j): # If i dominates j
                    dominated_sets[i].append(j)
                elif self.dominates(matrix_ret_risks, j, i): # If j dominates i
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
    
    def dominates(self, matrix_ret_risks, ind1, ind2):
        """
        Check if individual 1 dominates individual 2.
        
        Parameters:
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
            # Se puede hacer el evalute antes para evitar hacerlo dos veces

            front.sort(key=lambda x: self.evaluate(population[x])[m]) # Sort the front by the m-th objective
            distances[0] = distances[-1] = np.inf # Assign infinite distance to the limits
            min_val = self.evaluate(population[front[0]])[m]
            max_val = self.evaluate(population[front[-1]])[m]
            for i in range(1, len(front) - 1): # For each individual in the front (except the limits)
                if max_val - min_val == 0: # Avoid division by zero
                    distances[i] = 0
                else:
                    # Calculate the crowding distance, which is the normalized distance between the two neighbors
                    distances[i] += (self.evaluate(population[front[i + 1]])[m] -
                                 self.evaluate(population[front[i - 1]])[m]) / (max_val - min_val)
                 
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
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create one child.
        
        Parameters:
        - parent1: First parent (portfolio).
        - parent2: Second parent (portfolio).
        
        Returns:
        - child: Child (portfolio) created from the parents.

        """

        child = np.zeros(self.num_assets)
        parent1_indexes = np.where(parent1 > 0)[0]
        parent2_indexes = np.where(parent2 > 0)[0]

        equal_indexes = np.intersect1d(parent1_indexes, parent2_indexes)

        for index in equal_indexes:
            if random.random() > 0.5:
                child[index] = parent1[index]
            else:
                child[index] = parent2[index]
      
        parent1_indexes = np.setdiff1d(parent1_indexes, equal_indexes)
        parent2_indexes = np.setdiff1d(parent2_indexes, equal_indexes)

        for i in range(self.cardinality - len(equal_indexes)):
            if random.random() > 0.5:
                child[parent1_indexes[i]] = parent1[parent1_indexes[i]]
            else:
                child[parent2_indexes[i]] = parent2[parent2_indexes[i]]

        return child / child.sum()
    
    def mutation(self, individual):
        """
        Perform mutation on an individual.
        
        Parameters:
        - individual: Individual (portfolio) to mutate.
        
        Returns:
        - mutated_individual: Mutated individual (portfolio).
        
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] *= random.uniform(0.9, 1.1) # Mutation 10%
        individual /= individual.sum() # Normalize the mutated individual to sum to 1
        return individual

    def vary(self, population):
        """
        Apply genetic operations (crossover and mutation) to the population.

        Parameters:
        - population: The current population.

        Returns:
        - new_population: The new population after genetic operations.
        """
        new_population = []
        #fronts = self.fast_non_dominated_sort(population)
        for _ in range(self.N_pop):
            #parents = random.sample(fronts[0], 2) # Select two parents from the first front
            #child = self.crossover(population[parents[0]], population[parents[1]])
            parents = random.sample(list(population), 2) # Select two parents randomly
            child = self.crossover(parents[0], parents[1])
            child = self.mutation(child)
            new_population.append(child)
        return np.array(new_population)
    
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
            offsprings = self.vary(self.population_B)
            self.population_B = self.best(self.population_B, offsprings)
            i += 1            

        end_time = time.time()
        elapsed_time = end_time - started_time
        print(f"Execution time: {elapsed_time:.2f} seconds")
        return self.best(self.population_A, self.population_B)
    

    def plot_pareto_front(self, population):
        pareto_front = self.fast_non_dominated_sort(population)[0]
        pareto_points = np.array([self.evaluate(population[i]) for i in pareto_front])
        
        plt.figure(figsize=(8, 6))
        plt.scatter(pareto_points[:, 1], pareto_points[:, 0], color='red', label='Pareto Front')
        plt.xlabel('Variance')
        plt.ylabel('Mean')
        plt.title('Pareto front - Portfolio Optimization')
        plt.legend()
        plt.grid()
        plt.show()