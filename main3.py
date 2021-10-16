from deap import gp, tools, creator, base, algorithms
from sklearn.model_selection import train_test_split
from Source.code import *
import pandas as pd
import numpy as np
import itertools
import operator
import random

df = le_arq("./Data/telecom_users.csv")

train, test = train_test_split(df.to_numpy().tolist(), random_state=42)


