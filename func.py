from scipy.stats import bernoulli
import numpy as np
import networkx as nx
import random 


# Fitness calculation function call for a given genotype
# Genetic algorithm maximises this function
def calcobj(genotype, data):
    nodeselected = np.where(genotype == 1)
    sg = data.subgraph(nodeselected[0])
    val = nx.average_clustering(sg) - (nx.diameter(sg)/nx.number_of_nodes(sg)) #+ nx.sigma(sg)  # Function of properties of subgraph
    return val


# Boolean function for constraint checking
# Check for feasibility aka connectivity
def isFeasibleSolution(genotype, data):
    if np.count_nonzero(genotype) > 0:
        nodeselected = np.where(genotype == 1)
        sg = data.subgraph(nodeselected[0])
        return nx.is_connected(sg)
    else:
        return False


# Boolean function to check if "generateindividual" exists in "initialpopulation"
def listcompare(initialpopulation, generateindividual):
    if initialpopulation == []:
        return True
    else:
        for i in range(len(initialpopulation)):
            if np.all(initialpopulation[i] == generateindividual):
                return False
    return True


# Function to initialize population.
def initializepopulation(inpopulationsize, n, data):
    print("initializing population")
    initialpopulation = []
    while len(initialpopulation) < inpopulationsize:
        # generate individual
        generateindividual = bernoulli.rvs(0.4, size=n)  # Generates an n sized array, 0.01 probability of 1 bit
        # is generated individual valid?
        isfeasible = isFeasibleSolution(generateindividual, data)
        # if valid and does not already exist in population, add to population
        if isfeasible and listcompare(initialpopulation, generateindividual):
            initialpopulation.append(generateindividual)
    return initialpopulation


# Selection function. population1 is [[population], fitness].
def selection(population, popfitness):
    parents = []
    for _ in range(2):
        candidateparent = [random.randrange(0,len(population)) for _ in range(2)]  # Pick two candidate parents
        parent = np.argmax([popfitness[candidateparent[0]], popfitness[candidateparent[1]]])
        parent = population[candidateparent[parent]]
        parents.append(parent)

    return parents


# function to remove individual with lowest phenotypic value in case child added
def killindividual(popfitness):
    minpheno = 10**10
    presentindex = -1
    for i in range(len(popfitness)):
        if minpheno>popfitness[i]:
            minpheno = popfitness[i]
            presentindex = i
    return presentindex


# Function to perform GA
def ga(popsize, n, mutationrate, crossoverrate, numgen, data):
    print("GA started")
    # Set up GA
    objcallcount = 0  # To track number of fitness function calls

    # Define the initial population
    initialpopulation = initializepopulation(popsize, n, data)
    population = initialpopulation
    # population is a list of length population size and each element an n sized array
    popfitness = []  # Keeps the fitness values of the current population

    fitness = []  # Mean Fitness
    bestfitness = []  # Best Fitness
    medfitness = []  # Median Fitness

    for i in range(popsize):
        popfitness.append(calcobj(population[i], data))

    objcallcount += popsize
    bestfitness.append(max(popfitness))
    fitness.append(np.mean(popfitness))
    medfitness.append(np.median(popfitness))

    # Run GA
    for ii in range(numgen):
        # Selection:
        # List of list with each parent
        parentpair = selection(population, popfitness)  # Returns a pair of parents

        # Recombination and Mutation
        children = []

        # Crossover
        generatecrossoverflag = bernoulli.rvs(crossoverrate, size=1)
        if generatecrossoverflag[0] == 1:  # Do Crossover
            crossoverpoint = random.randrange(0, n, 1)
            children = [np.concatenate((parentpair[0][0:crossoverpoint], parentpair[1][crossoverpoint:int(n)]), axis=0),
                        np.concatenate((parentpair[1][0:crossoverpoint], parentpair[0][crossoverpoint:int(n)]), axis=0)]
        else:
            children = [parentpair[0], parentpair[1]]

        # Mutation
        genflag = bernoulli.rvs(mutationrate, size=1)
        if genflag[0] == 1:  # Do Mutation
            index = random.randrange(0, int(n), 1)
            for i in range(len(children)):
                children[i][index] = 1 - children[i][index]

        for j in range(len(children)):
            isfeasible = isFeasibleSolution(children[j], data)
            if isfeasible and listcompare(population, children[j]): # If feasible and not already present in population
                presentindex = killindividual(popfitness)
                population.pop(presentindex)
                popfitness.pop(presentindex)
                population.append(children[j])
                popfitness.append(calcobj(children[j], data))
                objcallcount += 1

        bestfitness.append(max(popfitness))
        fitness.append(np.mean(popfitness))
        medfitness.append(np.median(popfitness))

    retdict = {
            "population": population,
            "meanfitness": fitness,
            "medianfitness": medfitness,
            "bestfitness": bestfitness,
            "popfitness": popfitness
    }

    return retdict

