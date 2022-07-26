import sys
import time
import random
import statistics
import matplotlib.pyplot as plt

# CONSTANTS (weights in grams)
GOAL = 50000
NUM_RATS = 20
INITIAL_MIN_WT = 200
INITIAL_MAX_WT = 600
INITIAL_MODE_WT = 300
MUTATE_ODDS = 0.01
MUTATE_MIN = 0.5
MUTATE_MAX = 1.2
LITTER_SIZE = 8
LITTERS_PER_YEAR = 10
GENERATION_LIMIT = 1000

# ensure even-number of rats for breeding pairs:
if NUM_RATS % 2 != 0:
	NUM_RATS += 1

def populate(num_rats, min_wt, max_wt, mode_wt):
	"""Initialize a population with a triangular distribution of weights."""
	return [int(random.triangular(min_wt, max_wt, mode_wt))\
		for i in range(num_rats)]

def fitness(population, goal):
	"""Measure population fitness based on an attribute mean vs target."""
	ave = statistics.mean(population)
	return ave / goal

def select(population, to_retain):
	"""Cull a population to retain only a specified number of members."""
	sorted_population = sorted(population)
	to_retain_females = 16
	to_retain_males = 4
	females = sorted_population[:to_retain_females]
	males = sorted_population[to_retain_females:]
	selected_females = females[-to_retain_females:]
	selected_males = males[-to_retain_males:]
	return selected_males, selected_females

def breed(males, females, litter_size):
	"""Crossover genes among members (weights) of population."""
	random.shuffle(males)
	random.shuffle(females)
	children = []

	if len(females) % len(males) == 0:
		i = 0
		while i < len(females):
			for child in range(litter_size):
				child = random.randint(females[i], males[i % 4])
				children.append(child)
			i += 1
		return children
	else:
		print("Select desired number of females and males so that every male has the same number of females.")
		sys.exit()

def mutate(children, mutate_odds, mutate_min, mutate_max):
	"""Randomly alter rat weights using input odds & fractional changes."""
	for index, rat in enumerate(children):
		if mutate_odds >= random.random():
			children[index] = round(rat * random.uniform(mutate_min, mutate_max))
	return children

def main():
	"""Initialize population, select, bread, and mutate, display results."""
	generations = 0
	parents = populate(NUM_RATS, INITIAL_MIN_WT, INITIAL_MAX_WT, INITIAL_MODE_WT)
	print(f"initial population weights = {parents}")
	popl_fitness = fitness(parents, GOAL)
	print(f"initial population fitness = {popl_fitness}")
	print(f"number to retain = {NUM_RATS}")

	ave_wt = []
	max_wt = []

	while popl_fitness < 1  and generations < GENERATION_LIMIT:
		selected_males, selected_females = select(parents, NUM_RATS)
		children = breed(selected_males, selected_females, LITTER_SIZE)
		children = mutate(children, MUTATE_ODDS, MUTATE_MIN, MUTATE_MAX)
		parents = selected_males + selected_females + children
		popl_fitness = fitness(parents, GOAL)
		print("Generation {} fitness = {:.4f}".format(generations, popl_fitness))
		ave_wt.append(int(statistics.mean(parents)))
		max_wt.append(int(max(parents)))
		generations += 1
		
	print(f"average weight per generation = {ave_wt}")
	print(f"\nnumber of generations = {generations}")
	print(f"number of years = {int(generations / LITTERS_PER_YEAR)}")
	plt.plot(range(len(ave_wt)), ave_wt)
	plt.plot(range(len(max_wt)), max_wt)
	plt.show()
    

if __name__ == '__main__':
	start_time = time.time()
	main()
	end_time = time.time()
	duration = end_time - start_time
	print(f"\nRuntime for this program was {duration} seconds.")
