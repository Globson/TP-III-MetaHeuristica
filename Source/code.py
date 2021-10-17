import random
import operator
import itertools
import numpy as np
from deap import algorithms, base, creator, tools, gp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Source.leitura import *

df = le_arq("./Data/telecom_users.csv")
le = preprocessing.LabelEncoder()
for i in df:
    if(np.dtype(df[i]) == 'object'):
        df[i] = le.fit_transform(df[i])

customers=df.to_numpy().tolist()

train, test = train_test_split(customers, random_state=42)
# defined a new primitive set for strongly typed GP
pset = gp.PrimitiveSetTyped("MAIN", list(itertools.repeat(int, 17)) + list(itertools.repeat(float, 2)), bool, "IN")

# boolean operators
pset.addPrimitive(operator.and_, [bool, bool], bool)
pset.addPrimitive(operator.or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
# Define a protected division function
def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 1

pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
#pset.addTerminal(True, bool)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=5, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalcustomersbaseTrain(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Randomly sample 400 customers in the customers database
    train_samp = random.sample(train, 400)
    # Evaluate the sum of correctly identified customer as customers
    result = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in train_samp)
    return result, 

toolbox.register("evaluate", evalcustomersbaseTrain)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main(conf=True):
    random.seed()
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    if(conf):
        algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof, verbose=None)
    else:    
        algorithms.eaMuPlusLambda(pop, toolbox, 100, 100, 0.5, 0.1, 40, stats, halloffame=hof, verbose=None)
        
    return pop, stats, hof


'''
if __name__ == "__main__":
    for i in range(30):    
        _, _, tree = main()
        tree = tree[0]
        func = toolbox.compile(expr=tree)
        result_test = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in test) / len(test)
        result_train = sum(bool(func(*customer[:19])) is bool(customer[19]) for customer in train) / len(train)
        print(i,"-> Result Test: ", result_test)
        print(i,"-> Result Train: ",result_train)
        print(i,"-> Hof: ",tree)
'''
